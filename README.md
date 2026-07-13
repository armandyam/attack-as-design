# Attack-as-Design

## TL;DR

1. **Current paper.** *Attack-as-Design: Turning Surrogate Exploitation into Validated
   Biological Sequence Design.* LaTeX source and compiled PDF are in [`paper/`](paper/)
   ([`paper.pdf`](paper/paper.pdf)).
2. **Built on our earlier paper.** J. Krishnan, A. M. Rangarajan, A. Loehr and
   J. Hoelscher-Obermaier, "Adversarial Genomic Sequences Could Evade Biosecurity Screening,"
   *2026 IEEE Symposium on Security and Privacy Workshops (SPW)*, pp. 109-119.
   [doi:10.1109/SPW72489.2026.00013](https://doi.org/10.1109/SPW72489.2026.00013).
   That work built the attack. This work reuses the same search as a design tool.
3. **Sample run.** [`notebook/attack_as_design_pipeline.ipynb`](notebook/attack_as_design_pipeline.ipynb)
   runs the whole loop on a laptop in about ten seconds. No GPU, no large downloads.
   It is a small demonstration, not a full reproduction of the paper.

There is also a narrated explainer video:
[`video/attack_as_design_narrated.mp4`](video/attack_as_design_narrated.mp4).

## What this work does

We train a small model to predict a property of a biological sequence, then run an
adversarial-attack search to find sequences that model scores highly. We add a biological rule
that keeps the candidates realistic, and we check every design against signals the search never
used, so the result does not depend on the model grading its own work.

## Background

Sequence design is often posed as optimization. Train a model (the *surrogate*) to predict a
property such as binding affinity, thermostability, or activity, then search for sequences the
surrogate scores highly.

This has a known weakness. A surrogate is only accurate near the data it was trained on. When
the search moves into unusual sequences, the surrogate can score them highly by mistake, and the
search returns those mistakes. This is the same effect as an adversarial example in image
classification.

We use that weakness. The search that exploits a surrogate is the same machinery that generates
adversarial examples. If it can be stopped from drifting into unrealistic sequences, the same
search becomes a sequence designer. Two additions make this work:

1. **A biological rule that trains nothing.** Every candidate must use conservative BLOSUM62
   substitutions, and its whole-sequence properties (hydrophobicity, net charge, aromatic and
   charged fractions, sequence complexity, longest homopolymer run) must stay inside the range
   measured on natural sequences. Candidates outside the range are discarded during the search.

2. **Validation the search never saw.** Every design is checked against signals the optimizer
   never queried: held-out model ensembles, a separate thermostability predictor (TemStaPro), a
   structure predictor (AlphaFold2), known functional residues, an exact ground-truth oracle
   (TF-Bind-8), and a true DNA-activity model (Borzoi).

The system is demonstrated on four tasks across two molecule types: protein thermostability
(Meltome), GFP fluorescence, 600 bp regulatory DNA, and TF-Bind-8.

## Sample run

The notebook covers the full loop:

```
sequence -> features -> surrogate -> GA attack -> independent verification
```

```bash
pip install numpy scipy scikit-learn matplotlib
jupyter notebook notebook/attack_as_design_pipeline.ipynb   # then Run All
```

It runs on **TF-Bind-8 (SIX6)**, where all 65,536 possible 8-base-pair sequences have a measured
binding score. Every design can therefore be checked against an exact ground-truth oracle. The
notebook is self-contained and ships its own copy of the oracle (`notebook/y_SIX6.npy`, 256 KB).

**Scope.** The notebook shows the method on one of the four tasks. It does not rerun the protein
or regulatory-DNA experiments, the COMs and GABO comparisons, or the AlphaFold2, TemStaPro and
Borzoi validators.

## Repository layout

| Directory | Contents |
|---|---|
| `notebook/` | Self-contained pipeline notebook (TF-Bind-8, CPU, about 10 seconds). Start here. |
| `paper/`    | LaTeX source and compiled PDF. Figures are drawn in TikZ and also written individually to `paper/tikz-figures/`. |
| `code/`     | Reference implementation of the full study: embedding, surrogate training, optimizers, baselines, and validators. |
| `video/`    | Narrated explainer video. |

### Running the full study

`code/` is the reference implementation used for the paper. It is published so the method can be
read and reused. It does not run out of the box, because the large embedding matrices and the
per-run outputs are not included in this repository. To rerun the full study you need to
regenerate the embeddings with `code/embed_*.py` and `code/extract_evo2*.py` (GPU required), and
obtain the source datasets from their original providers:

- Meltome Atlas (protein thermostability), via FLIP.
- GFP deep mutational scanning (Sarkisyan et al. 2016), via ProteinGym.
- TF-Bind-8 (Barrera et al. 2016). The SIX6 oracle used by the notebook is included here.
- EPDnew human promoters for the regulatory-DNA task. Note EPDnew has academic-use terms.

## Results

- The search under the biological rule raises the true property on the design tasks. On 600 bp
  regulatory DNA it improved all 40 of 40 starting points, measured by a true activity model
  (Borzoi) that the search never queried.
- On thermostability, our rule matches a learned second model (GABO/aSCR) on naturalness (paired
  Wilcoxon p = 0.36) and beats a hand-tuned penalty (COMs) and the unconstrained search
  (p < 0.001).
- On GFP, our rule keeps 100% of designs inside the realistic biophysical range, against 5% to
  19% for the baselines. GFP numbers are reported over the 37 of 40 starting points that were
  realistic to begin with.
- The rule helps when it is tight, aligned with the objective, and the surrogate is exploitable.
  It can hurt when it is misaligned (WT1).
- The featurization is a swappable part of the system, reported as an ablation. A frozen
  foundation-model embedding helps only on long, variable-length protein. On short DNA a plain
  one-hot encoding is stronger. The system works with either.
- TF-Bind-8 is used as a diagnostic. Its landscape is saturated, so every method reaches roughly
  the same true score (top-16 around 0.89) and the task separates methods only by how much they
  overrate their own designs.

Full numbers and statistics are in [`paper/`](paper/).

## Models used

DNABERT-2-117M and ESM-2 650M, both frozen and mean-pooled. ProtT5 through TemStaPro.
AlphaFold2 through ColabFold 1.5.5. Borzoi and Evo2-7B as true-model validators and featurizers.
GPU jobs run on Modal A100.

## Citing

If you use this work, please cite the paper in [`paper/`](paper/) and our earlier IEEE SPW paper:

```bibtex
@inproceedings{krishnan2026adversarial,
  author    = {Krishnan, Jeyashree and Rangarajan, Ajay Mandyam and Loehr, Andrea
               and Hoelscher-Obermaier, Jason},
  title     = {Adversarial Genomic Sequences Could Evade Biosecurity Screening},
  booktitle = {2026 IEEE Symposium on Security and Privacy Workshops (SPW)},
  year      = {2026},
  pages     = {109--119},
  doi       = {10.1109/SPW72489.2026.00013}
}
```

## License

The code (`code/`, `notebook/`) is released under the MIT license. See [`LICENSE`](LICENSE).

The paper in `paper/` (text, LaTeX source, and figures) is not covered by the MIT license. It
remains copyright the authors and may become subject to a publisher's terms.

Datasets are not redistributed here. Obtain them from their original providers under their own
terms (see *Running the full study* above).
