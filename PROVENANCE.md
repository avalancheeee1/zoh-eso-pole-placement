# Provenance of the gain rule in Eq. (3)

This note exists because Reviewer 1 (comment 1) and Reviewer 3 (comment 1) asked for a
traceable source for the gain vector that the paper analyses, and because the answer is
partial: the formula is traceable, but its original source is not redistributable here.

## What the formula is

Eq. (3) of the manuscript is the innovation gain vector

    L = ( 3a , 3a^2 / Ts , 2a^3 / Ts^2 ),    a = 1 - rho,   rho = exp(-omega_o Ts)

and it is transcribed verbatim into `analysis_core.implemented_gains()`.

It is **not** taken from a textbook or from a published design rule. It is the gain formula
used by the reference implementation that motivated the study: a bandwidth-parameterized
third-order extended state observer written for a sampled double-integrator channel in a
marine-vehicle simulation environment. The accompanying documentation of that
implementation states the same intent as the gain rule above, namely a triple pole at
`rho`.

## Why the original implementation is not in this repository

The reference implementation is one component of a larger simulation system that is not
ours to release. It is therefore **not** redistributed here, and no version identifier for
it is published. What this repository does contain is the part that a reader can actually
check:

- the transcribed gain rule and its constants (`analysis_core.implemented_gains`), and
- `test_implemented_gain_rule_matches_the_transcribed_constants`, which pins those
  constants to the literal expression above so that the transcription cannot drift
  silently.

That is the honest boundary: the transcription is verifiable, the upstream source is not
available. Anyone who has the original implementation can confirm the transcription
directly against it.

## What the paper adds beyond the transcription

The paper's claim is **not** that a published design is faulty. It is that this
transcription rule — which is easy to copy into other discrete ESO codes, because it has
the familiar linear/quadratic/cubic structure in `a = 1 - rho` — does not perform the
triple-pole placement it states once the update timing is fixed. The contribution is the
exact characterization of the resulting polynomial defect and the minimal correction.

The cross-check against the published literature is in `README.md` and in the manuscript:
the third gain of Eq. (3) carries a spurious factor two. Neither the zero-order-hold
current-observer gains of Miklosovic et al. nor the standard discrete placement rule for a
triple integrator carries that factor; for the same predictor both give `(1 - rho)^3 / Ts^2`
in the third gain rather than `2 (1 - rho)^3 / Ts^2`.
