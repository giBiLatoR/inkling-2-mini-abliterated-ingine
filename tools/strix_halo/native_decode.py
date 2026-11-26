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
