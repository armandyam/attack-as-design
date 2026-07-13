# Attack-as-Design — reproducible pipeline notebook

A single, self-contained Jupyter notebook that reproduces the core loop of the paper
*"Attack-as-Design: Turning Surrogate Exploitation into Validated Biological Sequence Design."*

## What it does

```
sequence → features → surrogate → GA attack (with iterations) → independent verification
```

on **TF-Bind-8 (SIX6)** — the transcription-factor binding landscape of all 65,536 possible
8-base-pair DNA sequences. Because every sequence has a measured binding score, the notebook can
verify designs against an **exact ground-truth oracle** the surrogate never fully saw, plus a
disjoint held-out surrogate as a second independent check.

## Run it

```bash
pip install numpy scipy scikit-learn matplotlib
jupyter notebook attack_as_design_pipeline.ipynb   # then Run All
```

No GPU. No large downloads — the only data file, `y_SIX6.npy` (256 KB, the exact oracle), sits next
to the notebook and is also fetched automatically from the repo if missing. Executing every cell
takes about ten seconds on a laptop (plus a one-time `pip install` of the four standard packages).

## Files

- `attack_as_design_pipeline.ipynb` — the notebook (ships with all outputs and figures already rendered).
- `y_SIX6.npy` — the exact binding oracle: normalised affinity (0–1) for all 65,536 8-mers, in `ACGT`
  enumeration order.

## What you can change

- `FEAT` (section 3): `"onehot"` or `"kmer4"` — the featurization is a swappable component.
- The biological rule `g(x)` (section 4): a GC-content band and homopolymer-run cap, derived only from
  the training sequences (no extra model), that keeps designs realistic.
- `SEED` (section 0): reproducible randomness.
