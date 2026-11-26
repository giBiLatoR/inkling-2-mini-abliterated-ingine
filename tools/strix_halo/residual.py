"""Fused per-visit residual write from ingine's mix kernel for the gated detector checkpoint."""
import torch
import triton
import triton.language as tl

from inkling_ref import Ablation


@triton.jit
def _write(Y, DIR, DET, W, OFFSET, THRESHOLD, OUT, visit, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    valid = c < H
    y = tl.load(Y + row * H + c, valid, 0).to(tl.float32)
    r = tl.load(DIR + visit * H + c, valid, 0)
    d = tl.load(DET + visit * H + c, valid, 0)
    projection = tl.maximum(tl.sum(y * r, 0) - tl.load(OFFSET + visit), 0.)
    detect = tl.sum(y * d, 0) > tl.load(THRESHOLD + visit)
    delta = tl.load(W + visit) * tl.where(detect, projection, 0.) * r
    tl.store(OUT + row * H + c, y - delta, valid)


def residual_write(y, directions, detector, weights, offsets, thresholds, visit):
    hidden = y.shape[-1]
    src = y.contiguous()
    out = torch.empty_like(src)
    _write[(src.numel() // hidden,)](src, directions, detector, weights, offsets, thresholds,
                                    out, visit, hidden, triton.next_power_of_2(hidden), enable_fp_fusion=False)
    return out


class StrixAblation(Ablation):
    def _make_hook(self, kind):
        original = super()._make_hook(kind)
        def hook(module, args, output):
            if not self.enabled or not self.gate or self.detectors is None:
                return original(module, args, output)
            detector, thresholds = self.detectors[kind]
            return residual_write(output, getattr(self, kind + "_dirs"), detector,
                                  getattr(self, kind + "_w"), getattr(self, kind + "_c"),
                                  thresholds, self.visits.current)
        return hook
