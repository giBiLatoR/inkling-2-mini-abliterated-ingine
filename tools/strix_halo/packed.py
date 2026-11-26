"""Packed Inkling expert projections on AMD. Reuses the upstream model's router and activation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3fn(raw):
    exponent = (raw >> 3) & 15
    mantissa = (raw & 7).to(tl.float32)
    value = tl.where(exponent == 0, mantissa / 512.,
                     (1. + mantissa / 8.) * tl.exp2(exponent.to(tl.float32) - 7.))
    value = tl.where((raw & 127) == 127, float("nan"), value)
    return tl.where((raw & 128) != 0, -value, value)


@triton.jit
def _project(X, IDS, W, SCALE, FACTOR, Y, R: tl.constexpr, K: tl.constexpr,
             TOP: tl.constexpr, FP4: tl.constexpr, PER_ROUTE: tl.constexpr,
             ROWS: tl.constexpr, COLS: tl.constexpr):
    route = tl.program_id(1)
    expert = tl.load(IDS + route).to(tl.int64)
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, COLS)
    matrix_row = (expert * R + row[:, None])
    valid = (row[:, None] < R) & (col[None, :] < K)
    if FP4:
        pair = tl.load(W + matrix_row * (K // 2) + col[None, :] // 2, valid, 0)
        code = (pair >> (4 * (col[None, :] & 1))) & 15
        mag = code & 7
        val = tl.where(mag <= 4, mag * .5, tl.where(mag == 5, 3., tl.where(mag == 6, 4., 6.)))
        val = tl.where(code >= 8, -val, val)
        block = tl.load(SCALE + matrix_row * (K // 16) + col[None, :] // 16, valid, 0)
        weight = val * _e4m3fn(block) * tl.load(FACTOR + expert)
    else:
        raw = tl.load(W + matrix_row * K + col[None, :], valid, 0)
        block = tl.load(SCALE + matrix_row * (K // 32) + col[None, :] // 32, valid, 127)
        weight = _e4m3fn(raw) * tl.exp2(block.to(tl.float32) - 127.)
    source = route if PER_ROUTE else route // TOP
    x = tl.load(X + source * K + col, col < K, 0)
    acc = tl.sum(weight.to(tl.bfloat16).to(tl.float32) * x[None, :].to(tl.float32), 1)
    tl.store(Y + route * R + row, acc, row < R)


def project(experts, which, inputs, routes, per_route):
    weight = getattr(experts, which)
    scale = getattr(experts, which + "_scale")
    fp4 = experts.fmt == "nvfp4"
    rows = weight.shape[1]
    cols = inputs.shape[-1]
    output = inputs.new_empty((routes.numel(), rows))
    factor = getattr(experts, which + "_factor") if fp4 else inputs
    # On gfx1151 two rows measured faster for the wider MXFP8 gate/up projection.
    tile_rows = 2 if not fp4 and which == "w13" else 4
    _project[(triton.cdiv(rows, tile_rows), routes.numel())](
        inputs.contiguous(), routes.contiguous(), weight.view(torch.uint8), scale.view(torch.uint8),
        factor, output, rows, cols, routes.shape[1], fp4, per_route, tile_rows, triton.next_power_of_2(cols),
        num_warps=4)
    return output


def packed_forward(self, hidden_states, top_k_index, top_k_weights):
    gate_up = project(self, "w13", hidden_states, top_k_index, False)
    activations = self.act_fn(gate_up[:, 0::2], gate_up[:, 1::2])
    down = project(self, "w2", activations, top_k_index, True)
    weighted = down.view(*top_k_index.shape, -1) * top_k_weights[..., None]
    return weighted.float().sum(dim=1).to(hidden_states.dtype)
