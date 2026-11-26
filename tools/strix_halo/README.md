# Strix Halo single-request runner

This experimental CLI runs the upstream Inkling reference model on a Radeon 8060S (gfx1151). It requires the release ablation overlay on every run. The vision tower and projector come from the same checkpoint. The Triton kernel multiplies the routed experts directly from the packed weights, processing all selected routes together. A second Triton kernel fuses the per-visit ablation detector, dot products and residual update. The expert kernel covers NVFP4 in layers 0–19 and MXFP8 in layers 20–23. Attention, convolutions, dense layers and the vision tower currently run through PyTorch ROCm. DSpark and the native CUDA server are separate paths. Long-context capacity and sustained serving throughput on AMD have yet to be validated.

## Reproduce

Start with a Python 3.12 environment where ROCm PyTorch and Triton work on gfx1151. The measured setup used Torch 2.12.0a0+rocm7.13.0a20260411 and Triton 3.7.0. Keep the AMD Torch build installed when adding the Python packages below. The transformers commit is the one used by Nerv's reference.

```sh
python -m pip install 'transformers @ https://github.com/huggingface/transformers/archive/6173a2f679ff4ed393afc499aefc569da7e68bc1.tar.gz' accelerate safetensors numpy pillow
hf download satellitedown/Inkling-2-Mini-abliterated-ingine --revision cea73d67cd0e424c14c8b5cd276a9a2c2a5d7cbd --local-dir models/Inkling
(cd models/Inkling && sha256sum -c SHA256SUMS)
python tools/strix_halo/run.py models/Inkling 'Read the text in this image.' --image native/tests/golden/vision/fixtures/text_inkling42_320x80.png
python tools/strix_halo/run.py models/Inkling 'Count from 1 to 50.' --tokens 32
python tools/strix_halo/check_packed.py models/Inkling
python tools/strix_halo/check_residual.py models/Inkling
```

Add `--reference-experts` to compare the original dequantize-then-multiply experts and unfused ablation hooks on the same GPU. Both paths load the same detector, visit weights and directions and request a direct answer. The CLI writes the answer to stdout and timing to stderr. Time to first token begins immediately before generation, after model loading. Decode throughput counts the remaining token intervals and includes a generated stop token if present. The kernel comparison uses three warmups and 21 synchronized samples per projection. A new CLI invocation reloads the model. See [the on-device measurements](measurements.json).

## Native Radeon decode operators

With a gfx1151-capable ROCm HIP SDK, run `native/strix/build.sh` from the repository root. Set `ROCM_PATH` to the SDK root if it is outside `/opt/rocm`. The script compiles C++/HIP for gfx1151 and installs the library under `native/build-strix/`. The existing Strix CLI automatically uses it. Choose `--backend triton` to compare the previous implementation, or `--backend native` to require the build. Run `python tools/strix_halo/check_native.py` for operator parity.

The library implements batch-one four-tap causal convolution with history update and single-vector BF16/FP32 dense projections using one wave32 per output row. Prefill and unsupported shapes retain the previous operators. The JSON timing record includes native dispatch counts and p95 inter-token latency. [Matched NUC measurements](native-measurements.json) cover short-context decode.
