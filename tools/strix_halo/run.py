"""Single-request Strix Halo reference runner with Nerv's ablation and the checkpoint's vision tower."""
import argparse
import json
from time import perf_counter
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ablate
import inkling_ref as ref
from packed import packed_forward
from residual import StrixAblation


def image_inputs(path):
    with Image.open(path) as image:
        if ((image.width + 39) // 40) * ((image.height + 39) // 40) > 1024:
            raise ValueError("image exceeds the Strix prototype limit of 1024 tiles")
        rgba = np.asarray(image.convert("RGBA"), dtype=np.int32)
    height, width = rgba.shape[:2]
    y, x = np.ogrid[:height, :width]
    bg = np.where((y // 8 + x // 8) % 2 == 0, 204, 153)[..., None]
    alpha = rgba[:, :, 3:]
    rgb = (rgba[:, :, :3] * alpha + bg * (255 - alpha) + 127) // 255
    rows, cols = (height + 39) // 40, (width + 39) // 40
    pixels = np.full((rows * 40, cols * 40, 3), -1, dtype=np.float32)
    pixels[:height, :width] = rgb
    tiles = (pixels / 127.5 - 1).reshape(rows, 40, cols, 40, 3).transpose(0, 2, 1, 3, 4)
    return torch.from_numpy(tiles.reshape(-1, 40, 40, 3)).to(device="cuda", dtype=torch.bfloat16), rows, cols


# generate() sends the prompt once, then each generated token on the CPU.
class TokenClock:
    def __init__(self):
        self.steps = []

    def put(self, value):
        self.steps.append(perf_counter())

    def end(self):
        pass



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--reference-experts", action="store_true")
    parser.add_argument("--backend", choices=("auto", "native", "triton"), default="auto",
                        help="auto uses compiled gfx1151 operators when installed")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("requires a ROCm GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1151":
        parser.error("requires ROCm on gfx1151")
    overlay = ablate.load_ablation(args.model_dir / "inkling-2-mini-abliterated.safetensors")
    if not args.reference_experts:
        ref.PackedExperts.forward = packed_forward
    from native_decode import library_path, install_conv, install_matvec
    native = args.backend == "native" or (args.backend == "auto" and library_path().is_file()
                                          and not args.reference_experts)
    if native and args.reference_experts:
        parser.error("--reference-experts requires the reference backend")
    if native and not library_path().is_file():
        parser.error("build the gfx1151 operators with native/strix/build.sh first")
    conv_stats = install_conv() if native else {}
    model, tokenizer = ref.load(args.model_dir)
    gemv_stats = install_matvec(model) if native else {}
    visits = ref.Visits(model)
    hook = (ref.Ablation if args.reference_experts else StrixAblation)(model, visits)
    ablate.apply_ablation(SimpleNamespace(ablation=hook), overlay)
    parts, kwargs = [], {}
    if args.image:
        pixels, rows, cols = image_inputs(args.image)
        parts.append({"type": "image"})
        kwargs.update(pixel_values=pixels, image_grid_thw=torch.tensor([[1, rows, cols]], device="cuda"))
    parts.append({"type": "text", "text": args.prompt})
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": parts}], tokenize=False,
                                           reasoning_effort="none", add_generation_prompt=True)
    if args.image:
        marker = "<|unused_200054|>"
        if prompt.count(marker) != 1:
            raise ValueError("expected one image marker in rendered prompt")
        prompt = prompt.replace(marker, marker * (rows * cols))
    prompt += ablate.ANSWER_PREFIX
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    clock = TokenClock()
    ends = [tokenizer.convert_tokens_to_ids(t) for t in ablate.END_TOKENS]
    start = perf_counter()
    with torch.inference_mode():
        out = model.generate(**tokens, **kwargs, max_new_tokens=args.tokens, do_sample=False,
                             eos_token_id=ends, streamer=clock)
    times = clock.steps[1:]
    print(json.dumps({"backend": "native" if native else "reference" if args.reference_experts else "triton",
                      **conv_stats, **gemv_stats, "output_tokens": len(times), "ttft_s": times[0] - start,
                      "p95_intertoken_ms": float(np.percentile(np.diff(times), 95) * 1000) if len(times) > 1 else None,
                      "decode_tps": (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else None}),
          file=sys.stderr)
    print(tokenizer.decode(out[0, tokens.input_ids.size(1):], skip_special_tokens=True))


if __name__ == "__main__":
    main()
