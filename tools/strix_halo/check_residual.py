"""Verify the gated write on real attention and MLP overlay rows, including both detector outcomes."""
import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from residual import residual_write


def check(overlay, kind):
    device = "cuda"
    direction = F.normalize(overlay[kind + "_dirs"].to(device), dim=-1)
    detector, weight, offset, threshold = (overlay[kind + suffix].to(device) for suffix in
                                          ("_det", "_weight", "_offset", "_threshold"))
    visit = int(weight.abs().argmax())
    torch.manual_seed(29)
    inputs = torch.randn(32, direction.size(1), device=device) + 4 * direction[visit]
    # Put half the rows on either side of the detector threshold.
    target = threshold[visit] + torch.where(torch.arange(32, device=device) % 2 == 0, 3., -3.)
    inputs += ((target - inputs @ detector[visit]) / detector[visit].square().sum())[:, None] * detector[visit]
    y = inputs.bfloat16()
    v = y.float()
    passed = v @ detector[visit] > threshold[visit]
    assert passed.any() and (~passed).any()
    amount = (v @ direction[visit] - offset[visit]).clamp_min(0) * passed
    expected = (v - weight[visit] * amount[:, None] * direction[visit]).bfloat16()
    assert (expected != y).any(), "fixture must exercise an active intervention"
    actual = residual_write(y, direction, detector, weight, offset, threshold, visit)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=.008)
    assert torch.equal(actual[~passed], y[~passed])
    print(kind, "passed", "visit", visit, "detected", int(passed.sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    overlay = load_file(args.model_dir / "inkling-2-mini-abliterated.safetensors")
    for kind in ("attn", "mlp"):
        check(overlay, kind)
