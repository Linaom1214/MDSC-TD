# MDSC-TD

MDSC-TD: Memory-Driven Sparse Spatio-Temporal Interaction with Statistical Feature Calibration for Infrared Dim and Small Target Detection Under Strong Clutter

## Status

This repository currently provides the **model definition only**, released during the peer-review process. Training/evaluation scripts, dataset preparation code, and pretrained weights will be published here once the paper is accepted.

## Contents

- `model/MDSC_TD.py` — the proposed model (class `MDSC_TD`), used to produce the paper's main results.
- `model/MDSC_TD_ablation.py` — the ablation-configurable variant (class `MDSC_TD_Ablation`), exposing the module switches used in the paper's ablation studies.
- `model/memory/` — the memory-attention module shared by both, adapted from [SAM 2](https://github.com/facebookresearch/sam2) (Meta Platforms, Apache-2.0); original copyright headers are preserved.

Both model files are self-contained (each defines its own config helper and forward smoke test under `if __name__ == '__main__'`) and only depend on `model/memory/`.

## Forward signature

```python
pred, cur_memory = model(img, prev_memory, prev_mask)
```

`cur_memory` should be carried to the next frame; `prev_mask` is the previous frame's prediction for mask-guided variants. Reset both at the start of a new sequence.

## License

Apache License 2.0 (see `LICENSE`). Files under `model/memory/` retain their original Meta Platforms, Inc. copyright notice as required by that license.
