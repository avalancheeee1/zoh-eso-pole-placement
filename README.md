# Exact Pole Placement for a ZOH-Discretized Extended State Observer

Reference implementation and reproducible workflow for the paper

> **Exact Pole Placement for a ZOH-Discretized Extended State Observer: Noise Analysis and a JONSWAP Case Study**
> Xingda Li, Jianqiang Zhang, Bo Zhang, Pengfei Zhang, Ling Tan
> MDPI *Mathematics* (under revision)

This repository is the code and data release requested in review (Reviewer 1, comment 12;
Reviewer 3, comment 6). It regenerates every table, every figure and every numerical value
reported in the manuscript.

**Release used for the first revision:** tag `v1.0.0`. The manuscript's Data Availability
Statement cites this repository as the exact version used for the revision; that version is
this tag. (The title above is the revised title; earlier drafts of the paper used
"Noise-Aware Auditing", and this repository was first published under that wording.)

## What the code shows

A third-order extended state observer (ESO) that combines an exact zero-order-hold state
predictor with a bandwidth-parameterized innovation gain vector. The bandwidth
parameterization is the familiar one — gains linear, quadratic and cubic in `a = 1 - rho`
with `rho = exp(-omega_o Ts)` — but it does **not** place a triple pole at `rho` once the
update timing is fixed. The characteristic polynomial of the implemented update carries a
correction proportional to the third gain,

    (1 - rho)^3 * lambda        (the "defect")

so the implemented spectral radius returns towards one as the requested bandwidth grows,
even though the observer stays Schur stable at every positive bandwidth. Matching the
characteristic polynomial term by term gives a short corrected gain vector that does place
all three poles at `rho` for the same update timing.

Two further results are in the code: the correct input channel for a sample-to-sample
disturbance increment (which is *not* the same as the extended-state channel for a constant
disturbance), and the separation of three noise measures that are often used loosely —
the white-noise variance factor, the unit-impulse peak, and the induced bounded-input gain.

## Repository layout

| Path | Contents |
|------|----------|
| `analysis_core.py` | The verified kernel: gains, error matrices, noise metrics, JONSWAP spectrum, tuning criterion. No I/O, no plotting. |
| `validate_theory.py` | Runs the four numerical experiments, writes `generated_data/*.csv` and `generated_data/summary.json`, then calls the figure scripts. |
| `test_validation.py` | The regression suite (11 tests). Guards every analytic claim in the paper. |
| `redraw_advanced_figures.py` | Renders the manuscript figures from the generated CSVs. |
| `simulate_closed_loop.py` | The closed-loop ADRC simulation (manuscript Figure 3). |
| `rev_robustness.py` | The robustness study added in revision (manuscript Figure 4 and Table 3). |
| `generated_data/` | Output CSVs and JSON. Committed, so results can be inspected without running anything. |
| `figures/` | Output figures (PDF vector, SVG, 400 dpi PNG, 600 dpi LZW TIFF, grayscale preview). |
| `RESULTS.md`, `RESULTS_raw.txt` | Readable and raw dumps of the run, also written by `validate_theory.py`. |
| `FIGURE_CONTRACTS.md` | The visual specification each figure is held to, written by `redraw_advanced_figures.py`. |
| `PROVENANCE.md` | What the analysed gain rule is, and why its upstream source is not redistributed here. |

## Reproduce

```bash
python -m pip install -r requirements.txt

python -m pytest -q test_validation.py   # regression suite, ~1 s
python validate_theory.py                # data + Figures 1, 2, 5, 6
python simulate_closed_loop.py           # Figure 3
python rev_robustness.py                 # Figure 4 and Table 3 evidence
```

`validate_theory.py` takes a few minutes: the Monte Carlo sampling experiment dominates the
runtime. Everything else finishes in seconds.

The workflow is deterministic. The committed CSVs and `summary.json` in `generated_data/`
are reproduced byte-for-byte by a fresh run on the pinned dependency set; no random seed is
left to the environment (`analysis_core.SEED = 42`).

## Manuscript cross-reference

Figures:

| Manuscript | Files |
|------------|-------|
| Figure 1 — Pole-placement audit | `figures/fig_pole_audit.*` |
| Figure 2 — Norm-consistent noise comparison | `figures/fig_norm_consistent_noise.*` |
| Figure 3 — Closed-loop disturbance rejection | `figures/fig_closed_loop.*` |
| Figure 4 — Robustness checks *(added in revision)* | `figures/fig_rev_robustness.*` |
| Figure 5 — Monte Carlo convergence | `figures/fig_corrected_sampling.*` |
| Figure 6 — Bandwidth-selection evidence | `figures/fig_jonswap_case_tuning.*` |

Tables:

| Manuscript | Files |
|------------|-------|
| Table 1 — Pole audit | `generated_data/pole_audit.csv` |
| Table 2 — Noise metrics | `generated_data/noise_metrics.csv` |
| Table 3 — Closed-loop robustness *(added in revision)* | `generated_data/closed_loop_robustness.csv` |
| Table 4 — JONSWAP selection | `generated_data/jonswap_case_tuning.csv`, `generated_data/tuning_sensitivity.csv` |

The full machine-readable output of the robustness study, including the matched
spectral-radius comparison, the `Ts`-invariance check and the four Jury conditions, is in
`generated_data/rev_robustness.json`.

## Scope and honest limitations

These are stated in the manuscript and repeated here so the code is not read as claiming
more than it does.

- **The JONSWAP spectrum is used as a normalized disturbance-spectrum shape, not as a
  validated vessel load spectrum.** No wave-to-force or wave-to-acceleration transfer
  function is derived or assumed, no disturbance scale is identified from sea-trial data,
  and the bandwidths selected in the case study are illustrative design examples, not
  physically validated marine-control settings.
- **The case-study parameters are declared scenario assumptions**, not identified
  hydrodynamic parameters.
- **The robustness study covers measurement noise and controller bandwidth.** It does not
  cover sampling jitter, actuator delay, quantization or unmodelled plant dynamics. Each of
  those requires either a re-derivation of the gains or a separate analysis.
- **The regression suite is an internal-consistency check, not an independent proof of
  correctness.** It fails if an implemented formula drifts from the analytic expression it
  is supposed to match; it cannot detect an error that is present in both.
- **Monotonicity of the spectral-radius rebound at large bandwidth is a numerical
  observation**, verified over the bandwidth range reported in the paper and located to
  high precision near its interior minimum. No closed-form monotonicity proof is offered.
- **No result here proves stability of a nonlinear multi-vessel controller.** The
  closed-loop analysis is a nominal-channel interpretation.

## Environment

Developed and tested on CPython 3.13 with

    numpy 2.2.3, scipy 1.16.1, matplotlib 3.10.8, pytest 9.0.3

## License

MIT — see `LICENSE`.
