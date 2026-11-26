"""Dispatch supported single-token convolution shapes to the native gfx1151 build."""
import ctypes
from pathlib import Path

import torch
from transformers.models.inkling2_loop import modeling_inkling2_loop as inkling


def library_path():
    return Path(__file__).resolve().parents[2] / "native/build-strix/libingine_strix.so"


def install_conv():
    kernel = ctypes.CDLL(str(library_path())).strix_conv
    ptr = ctypes.c_void_p
    kernel.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]
    kernel.restype = ctypes.c_int
    original = inkling.Inkling2LoopShortConvolution.forward
    counts = {"native_conv_calls": 0}

    def forward(self, hidden_states, dynamic_input=None, cache=None, conv_mask=None, cache_offset=0, **kwargs):
        idx = self.layer_idx + cache_offset
        state = cache.get_conv_state(idx, state_idx=self.conv_idx) if cache is not None else None
        w = self.conv1d.weight
        c = hidden_states.shape[-1]
        allowed = (torch.float32, torch.bfloat16)
        if (hidden_states.shape == (1, 1, c) and conv_mask is None and state is not None
                and not cache.is_recording_past(idx) and self.conv1d.bias is None
                and state.shape == (1, c, 4) and w.shape == (c, 1, 4)
                and all(t.is_contiguous() and t.dtype in allowed and t.device == hidden_states.device
                        for t in (hidden_states, state, w))):
            out = torch.empty_like(hidden_states)
            rc = kernel(hidden_states.data_ptr(), state.data_ptr(), w.data_ptr(), out.data_ptr(), c,
                        hidden_states.dtype == torch.float32, state.dtype == torch.float32,
                        w.dtype == torch.float32, torch.cuda.current_stream().cuda_stream)
            if rc:
                raise RuntimeError(f"gfx1151 convolution launch failed with HIP error {rc}")
            counts["native_conv_calls"] += 1
            return out
        return original(self, hidden_states, dynamic_input=dynamic_input, cache=cache,
                        conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)

    inkling.Inkling2LoopShortConvolution.forward = forward
    return counts


def install_matvec(model):
    op = ctypes.CDLL(str(library_path())).strix_matvec
    ptr = ctypes.c_void_p
    op.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]
    op.restype = ctypes.c_int
    counts = {"native_gemv_calls": 0}

    def bind(module):
        original = module.forward
        def forward(x):
            w = module.weight
            if (x.ndim == 0 or x.shape[-1] != w.size(1) or x.numel() != w.size(1) or not x.is_contiguous() or x.dtype != w.dtype
                    or x.device != w.device or x.data_ptr() % (4 * x.element_size())):
                return original(x)
            out = x.new_empty((*x.shape[:-1], w.size(0)))
            rc = op(x.data_ptr(), w.data_ptr(), out.data_ptr(), *w.shape, w.dtype == torch.float32,
                    torch.cuda.current_stream().cuda_stream)
            if rc:
                raise RuntimeError(f"gfx1151 matvec launch failed with HIP error {rc}")
            counts["native_gemv_calls"] += 1
            return out
        module.forward = forward
    for module in model.modules():
        if (type(module) is torch.nn.Linear and module.bias is None and module.weight.is_cuda
                and module.weight.dtype in (torch.bfloat16, torch.float32) and module.weight.is_contiguous()
                and module.in_features % 4 == 0 and module.weight.data_ptr() % (4 * module.weight.element_size()) == 0):
            bind(module)
    return counts


def install_experts():
    import packed
    op = ctypes.CDLL(str(library_path())).strix_experts
    ptr = ctypes.c_void_p
    op.argtypes = [ptr]*6 + [ctypes.c_int]*6 + [ptr]
    op.restype = ctypes.c_int
    stats = {"native_expert_calls": 0}

    def project(experts, which, inputs, ids, per_route):
        fp4 = experts.fmt == "nvfp4"
        w = getattr(experts, which)
        scale = getattr(experts, which + "_scale")
        factor = getattr(experts, which + "_factor") if fp4 else None
        rows, cols, top = w.shape[1], inputs.size(-1), ids.size(1)
        if inputs.dtype != torch.bfloat16 or ids.dtype != torch.long or cols % 32:
            raise ValueError("unsupported packed Inkling expert layout")
        x, routes = inputs.contiguous(), ids.contiguous()
        output = x.new_empty((routes.numel(), rows))
        for start in range(0, ids.shape[0], 65535 // top):
            count = min(65535 // top, ids.shape[0] - start) * top
            offset = start * top if per_route else start
            rc = op(x[offset].data_ptr(), routes[start].data_ptr(), w.data_ptr(), scale.data_ptr(),
                    factor.data_ptr() if fp4 else None, output[start * top].data_ptr(),
                    rows, cols, top, count, fp4, per_route, torch.cuda.current_stream().cuda_stream)
            if rc:
                raise RuntimeError(f"native expert launch failed: HIP {rc}")
            stats["native_expert_calls"] += 1
        return output

    lib = ctypes.CDLL(str(library_path()))
    activate, combine = lib.strix_activate, lib.strix_combine
    activate.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ptr]
    combine.argtypes = [ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ptr]

    def forward(experts, hidden, ids, weights):
        up = project(experts, "w13", hidden, ids, False)
        mid = up.new_empty((up.size(0), experts.intermediate_dim))
        stream = torch.cuda.current_stream().cuda_stream
        rc = activate(up.data_ptr(), mid.data_ptr(), mid.numel(),
                      experts.act_fn.gate_softcap.softcap, experts.act_fn.up_softcap.softcap, stream)
        if rc:
            raise RuntimeError(f"native expert activation failed: HIP {rc}")
        down = project(experts, "w2", mid, ids, True)
        out = hidden.new_empty(hidden.shape)
        weights = weights.float().contiguous()
        rc = combine(down.data_ptr(), weights.data_ptr(), out.data_ptr(), out.numel(), hidden.size(-1),
                     ids.size(1), stream)
        if rc:
            raise RuntimeError(f"native expert reduction failed: HIP {rc}")
        return out

    from inkling_ref import PackedExperts
    PackedExperts.forward = forward
    packed.project = project
    return stats
