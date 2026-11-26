"""Compare the compiled gfx1151 decode operations against torch on the device."""
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from native_decode import install_conv, install_matvec, inkling


def check_linear():
    for dtype in (torch.bfloat16, torch.float32):
        for cols, rows in ((2304, 3072), (4608, 64), (128, 31)):
            op = torch.nn.Linear(cols, rows, bias=False, device="cuda", dtype=dtype)
            x = torch.randn(1, 1, cols, device="cuda", dtype=dtype)
            expected = op(x)
            calls = install_matvec(op)
            actual = op(x)
            torch.testing.assert_close(actual, expected, atol=.008 if dtype == torch.bfloat16 else 2e-5,
                                       rtol=.008 if dtype == torch.bfloat16 else .0001)
            assert calls["native_gemv_calls"] == 1
            multi = x.expand(1, 2, cols)
            torch.testing.assert_close(op(multi), F.linear(multi, op.weight))
            assert calls["native_gemv_calls"] == 1


def check_conv():
    op = inkling.Inkling2LoopShortConvolution(128, 4, 0, 0).to("cuda", torch.bfloat16)
    state = torch.randn(1, 128, 4, device="cuda").bfloat16().float()
    history = state.clone()
    cache = SimpleNamespace(get_conv_state=lambda *_, **kw: state, is_recording_past=lambda _: False)
    calls = install_conv()
    for i in range(5):
        x = torch.randn((1, 1, 128), device="cuda").bfloat16()
        history = torch.cat((history[:, :, 1:], x.transpose(1, 2).float()), -1)
        dot = F.conv1d(history.bfloat16(), op.conv1d.weight, groups=128).transpose(1, 2)
        expect = ((dot.float() + x.float()) / (5 ** .5)).bfloat16()
        torch.testing.assert_close(op(x, cache=cache), expect, rtol=.008, atol=.008)
        torch.testing.assert_close(state, history)
    assert calls["native_conv_calls"] == 5


if __name__ == "__main__":
    torch.manual_seed(41)
    with torch.inference_mode():
        check_linear()
        check_conv()
    print("gfx1151 GEMV, fallback and five-step convolution parity passed")
