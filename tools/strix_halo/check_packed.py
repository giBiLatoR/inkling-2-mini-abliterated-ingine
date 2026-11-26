"""Check both packed projections against Nerv's BF16 reference on real checkpoint rows."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace

from safetensors import safe_open
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling_ref import E2M1, PackedExperts
import packed
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopSoftcappedSwiGLU


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(21):
        start = perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples), "p95_ms": sorted(samples)[19]}


def check(f, layer, which):
    prefix = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
    def read(suffix):
        return f.get_tensor(prefix + suffix)[:8].to("cuda")
    fp4 = layer == 0
    obj = SimpleNamespace(fmt="nvfp4" if fp4 else "mxfp8")
    weight = read("")
    setattr(obj, which, weight)
    setattr(obj, which + "_scale", read("_scale"))
    if fp4:
        setattr(obj, which + "_factor", read("_amax") / read("_outer_scale_denominator"))
        obj.lut = torch.tensor(E2M1, device="cuda")
    cols = weight.shape[2] * (2 if fp4 else 1)
    ids = torch.tensor([[6, 2, 7, 1, 0, 4, 3, 5]], device="cuda")
    x = torch.randn((8 if which == "w2" else 1, cols), device="cuda", dtype=torch.bfloat16)
    def candidate():
        return packed.project(obj, which, x, ids, which == "w2")
    order = ids[0].tolist()
    def oracle():
        return torch.stack([F.linear(x[i if which == "w2" else 0],
                                     PackedExperts.decode(obj, which, expert)) for i, expert in enumerate(order)])
    baseline = oracle().float()
    actual = candidate().float()
    rel = ((actual - baseline).norm() / baseline.norm()).item()
    if not rel < .005:
        raise AssertionError(f"layer={layer}, {which}: relative RMS {rel}")
    print(json.dumps({"layer": layer, "projection": which, "relative_rms": rel,
                      "reference": timed(oracle), "packed": timed(candidate)}), flush=True)


def check_native_block(f, layer):
    # Repeated experts across three tokens exercise route offsets and weighted reduction.
    fp4 = layer == 0
    obj = SimpleNamespace(fmt="nvfp4" if fp4 else "mxfp8", intermediate_dim=1152,
                          act_fn=Inkling2LoopSoftcappedSwiGLU(SimpleNamespace(mlp_gate_softcap=4., mlp_up_softcap=25.)))
    obj.lut = torch.tensor(E2M1, device="cuda")
    for which in ("w13", "w2"):
        prefix = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
        def read(suffix):
            return f.get_tensor(prefix + suffix)[:8].to("cuda")
        setattr(obj, which, read(""))
        setattr(obj, which + "_scale", read("_scale"))
        if fp4:
            setattr(obj, which + "_factor", read("_amax") / read("_outer_scale_denominator"))
    x = torch.randn(3, 2304, dtype=torch.bfloat16, device="cuda")
    routes = torch.stack([torch.randperm(8, device="cuda") for _ in range(3)])
    weights = torch.randn(3, 8, device="cuda").softmax(-1) * 8
    expected = torch.zeros_like(x, dtype=torch.float32)
    for j in range(8):
        up = F.linear(x, PackedExperts.decode(obj, "w13", j))
        mid = obj.act_fn(up[:, 0::2], up[:, 1::2])
        down = F.linear(mid, PackedExperts.decode(obj, "w2", j))
        expected += down * weights.gather(1, (routes == j).int().argmax(1, keepdim=True))
    actual = PackedExperts.forward(obj, x, routes, weights).float()
    rel = ((actual-expected.bfloat16()).norm() / expected.norm()).item()
    if not rel < .005:
        raise AssertionError(f"whole block layer {layer}: relative RMS {rel}")
    print(json.dumps({"layer": layer, "whole_expert_block_relative_rms": rel}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.native:
        from native_decode import install_experts
        install_experts()
    torch.manual_seed(41)
    with safe_open(args.model_dir / "model-00001-of-00001.safetensors", framework="pt") as checkpoint:
        for layer in (0, 20):
            for which in ("w13", "w2"):
                check(checkpoint, layer, which)
            if args.native:
                check_native_block(checkpoint, layer)
