# Strix Halo single-request runner

This experimental CLI runs the upstream Inkling reference model on a Radeon 8060S (gfx1151). It requires the release ablation overlay on every run. The vision tower and projector come from the same checkpoint. The compiled AMD backend handles routed experts (NVFP4 in layers 0–19 and MXFP8 in layers 20–23), eligible dense matrix-vector products, and short decode convolutions. Triton handles the fused ablation update and provides a fallback for expert projections. Attention, the vision tower and other operators run through PyTorch ROCm. Greedy DSpark uses the same target and vision path. Upstream's CUDA server is separate. Long-context capacity and sustained serving throughput on AMD have yet to be validated.

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

Add `--reference-experts` to compare the original dequantize-then-multiply experts and unfused ablation hooks on the same GPU. Use `--dspark 0` for one-token-at-a-time comparisons. Both paths load the same detector, visit weights and directions and request a direct answer. The CLI writes the answer to stdout and timing to stderr. Time to first token begins immediately before generation, after model loading. Decode throughput counts the remaining token intervals and includes a generated stop token if present. The kernel comparison uses three warmups and 21 synchronized samples per projection. A new CLI invocation reloads the model. See [the on-device measurements](measurements.json).

## Native Radeon decode operators

With a gfx1151-capable ROCm HIP SDK, run `native/strix/build.sh` from the repository root. Set `ROCM_PATH` to the SDK root if it is outside `/opt/rocm`. The script compiles C++/HIP for gfx1151 and installs the library under `native/build-strix/`. The existing Strix CLI automatically uses it. Choose `--backend triton` to compare the previous implementation, or `--backend native` to require the build. Run `python tools/strix_halo/check_native.py` for operator parity.

The library implements batch-one four-tap causal convolution with history update and single-vector BF16/FP32 dense projections using one wave32 per output row. Prefill and unsupported shapes retain the previous operators. The JSON timing record includes native dispatch counts and p95 inter-token latency. [Matched NUC measurements](native-measurements.json) cover short-context decode.


## Native experts and greedy DSpark

Rebuild with `native/strix/build.sh` after updating the fork. The compiled backend runs NVFP4 and MXFP8 expert matrix-vector products, softcapped SwiGLU and weighted route reduction directly on gfx1151. The native CLI defaults to four draft guesses. Use `--dspark 0` for ordinary greedy decoding, or `--dspark 1` through `--dspark 7` to select the maximum guesses per verification. A native build from before the expert port needs recompilation.

This ROCm implementation loads the released `mtp.safetensors`, fuses target features from visits 5, 23, 41 and the final normalized hidden state, and applies the three DSpark attention/MLP layers and recurrent Markov head. Draft attention and projections use BF16. The shortlist uses upstream frequency-ranked IDs, special tokens and input IDs. Each verification batches the anchor and proposed tokens through the abliterated target. Only its matching prefix and next prediction can be emitted. Rejected KV, convolution history and draft features are excluded from the next round. Greedy is the only sampling mode in this CLI. Batch and single-row numerical paths can pick different tokens, so exact serial output is workload dependent.

```sh
python tools/strix_halo/check_packed.py models/Inkling --native
python tools/strix_halo/check_draft.py models/Inkling
python tools/strix_halo/run.py models/Inkling 'Write a Python interval merger.' --tokens 256
```

The timing record includes the depth, verification rounds, proposed/accepted guesses, and a digest of output token IDs. Tokens verified together become available in bursts; the inter-token p95 includes the gaps between verification rounds. First-token latency includes drafter setup. See [AMD expert and DSpark results](dspark-measurements.json).
