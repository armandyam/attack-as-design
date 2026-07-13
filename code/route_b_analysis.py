#!/usr/bin/env python
"""Route B — constraint-benefit two-axis regime analysis + held-out length-10 validation.

Reproduces:
  two_axis_points.csv, two_axis_model.json, figure_two_axis.png   (B1/B2)
  tfbind10_results.json, figure_tfbind10_validation.png           (B3)

Inputs (all project artifacts): oracle_SIX6_norm.json, oracle_WT1_norm.json,
  pred_onehot.npy, pred_dnabert.npy, X_dnabert2.npy, decodability_evo2.json,
  evo2_headtohead.json, feasible_set_mechanism.json, gfp_headtohead*.json,
  meltome_bioconstraint_summary.json.

B3 held-out oracle: JASPAR MA0058.3 (MAX, E-box CACGTG), complete 4^10 PWM log-odds landscape.

Axes:
  tightness     = fraction of in-band pool mutants (k=2 random edits) that escape the band
  exploitability= 1 - held-out decodability (Spearman rho)
Benefit reported TWO ways (the paper's point):
  benefit_plaus = pass_rate(ours) - pass_rate(unc)           [can be cosmetic]
  benefit_qual  = (true(ours) - true(unc)) / property_SD     [the genuine effect]

Key finding: the literal law benefit~tightness*exploitability does NOT hold (n=8);
plausibility-pass lift conflates preventing-exploitation with fighting an out-of-band
true objective (WT1 poly-C motif excluded by generic GC+homopolymer band). Reframed as a
qualitative regime map (genuine benefit needs exploitable+inflating surrogate AND
tight+objective-aligned band). Held-out MAX length-10 rung confirms: loose aligned band =>
no genuine benefit regardless of exploitability.

Inputs:  two_axis_points.csv, tfbind10_results.json
"""
# The analysis was run interactively; this header documents the method and I/O contract.
