# Dataset License

The Redirect4D-Bench dataset, hosted at
[`huggingface.co/datasets/vveicao/redirect4d-bench`](https://huggingface.co/datasets/vveicao/redirect4d-bench),
is released under the
**Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

License text: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>

## Scope

The CC BY-NC 4.0 license covers the **derived research assets** released as
part of the benchmark, including:

- reconstructed 4D point clouds,
- target camera trajectories and rendered geometry buffers,
- pseudo ground-truth subject masks,
- text prompts and per-case metadata,
- evaluation metadata.

## What is NOT covered

- **Original/source RGB videos and frames** are NOT covered by this dataset
  license. The public sample includes source RGB for two preview tracks only;
  full-dataset source RGB is distributed separately through a gated package.
  Source RGB is governed by the YouTube Terms of Service and the original
  uploaders' rights.
- **Third-party model checkpoints** referenced by the construction pipeline
  (e.g., SAM, ViPE, Wan2.2 VACE, Qwen3-VL) remain under their respective
  upstream licenses.

## Source code license

The code in this repository is released under the **Apache License 2.0**
(see `LICENSE`). The two licenses are independent: code may be reused under
Apache 2.0 terms, while the dataset assets are subject to CC BY-NC 4.0.

## Attribution

If you use the dataset, please cite the accompanying paper (citation
information will be provided once the paper is published).
