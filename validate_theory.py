"""Reproducible validation for the revised ZOH-ESO manuscript.

Run ``python validate_theory.py`` to regenerate all tables, figures, CSV
files, JSON metadata, and the Markdown results summary from one code path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_core import *  # re-export the verified core for tests/users

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
DATA = ROOT / "generated_data"
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def _style(axis, log_grid: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, which="both" if log_grid else "major", alpha=0.22, lw=0.4)


def _save_figure(figure, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / f"{stem}.pdf")
    figure.savefig(FIGURES / f"{stem}.png")
    plt.close(figure)


def _write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (DATA / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment_poles() -> list[dict]:
    rows = []
    products = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    for product in products:
        omega_o = product / DT
        rho = np.exp(-product)
        implemented = implemented_error_matrix(omega_o, DT)
        exact = exact_error_matrix(omega_o, DT)
        polynomial_error = float(np.max(np.abs(np.poly(implemented) - np.poly([rho] * 3))))
        rows.append({
            "omega_o_Ts": product,
            "rho": rho,
            "implemented_spectral_radius": np.max(np.abs(np.linalg.eigvals(implemented))),
            "exact_spectral_radius": np.max(np.abs(np.linalg.eigvals(exact))),
            "polynomial_coefficient_error": polynomial_error,
        })
    _write_csv("pole_audit.csv", rows)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    axes[0].semilogx(products, [r["implemented_spectral_radius"] for r in rows], "o-", color=COLORS[0], label="implemented gains")
    axes[0].semilogx(products, np.exp(-products), "--", color=COLORS[1], label="target pole")
    axes[0].axhline(1.0, color="black", ls=":", lw=0.8)
    axes[0].set_xlabel(r"$\omega_oT_s$")
    axes[0].set_ylabel("spectral radius")
    axes[0].legend(frameon=False)
    _style(axes[0])
    axes[1].loglog(products, [r["polynomial_coefficient_error"] for r in rows], "o-", color=COLORS[1], label="computed")
    axes[1].loglog(products, (1.0 - np.exp(-products)) ** 3, "k--", label=r"$(1-\rho)^3$")
    axes[1].set_xlabel(r"$\omega_oT_s$")
    axes[1].set_ylabel("coefficient error")
    axes[1].legend(frameon=False)
    _style(axes[1], log_grid=True)
    _save_figure(figure, "fig_pole_audit")
    return rows


def experiment_sampling() -> tuple[list[dict], float]:
    omega_p = 2.0 * np.pi / TP_TABLE[3]
    target = physical_spectral_moment(omega_p, 0.6, 2.4)
    sample_sizes = [8, 16, 32, 64, 128, 256, 512]
    rows = []
    for sample_size in sample_sizes:
        estimates = []
        for trial in range(400):
            components = generate_wave_components(3, N=sample_size, seed=10_000 + trial)
            estimates.append(float(np.sum(components["amp_raw"] ** 2) / 2.0))
        estimates_array = np.asarray(estimates)
        relative = (estimates_array - target) / target
        rows.append({
            "N": sample_size,
            "mean_signed_relative_error": float(np.mean(relative)),
            "mean_absolute_relative_error": float(np.mean(np.abs(relative))),
            "standard_deviation_relative_error": float(np.std(relative, ddof=1)),
            "standard_error_mean": float(np.std(relative, ddof=1) / np.sqrt(len(relative))),
        })
    _write_csv("spectral_sampling.csv", rows)
    share = physical_band_share(omega_p)

    figure, axis = plt.subplots(figsize=(3.4, 2.5))
    axis.loglog(sample_sizes, [max(abs(r["mean_signed_relative_error"]), 1e-4) for r in rows], "o-", color=COLORS[1], label="absolute bias")
    axis.loglog(sample_sizes, [r["standard_deviation_relative_error"] for r in rows], "s--", color=COLORS[0], label="standard deviation")
    axis.loglog(sample_sizes, 1.5 / np.sqrt(sample_sizes), "k:", label=r"$N^{-1/2}$")
    axis.set_xlabel("number of components N")
    axis.set_ylabel("relative sampling error")
    axis.legend(frameon=False)
    _style(axis, log_grid=True)
    _save_figure(figure, "fig_corrected_sampling")
    return rows, share


def experiment_noise() -> list[dict]:
    bandwidths = [1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 30.0]
    rows = []
    for design in ("implemented", "exact"):
        for omega_o in bandwidths:
            rows.append({
                "design": design,
                "omega_o": omega_o,
                **noise_metrics(omega_o, DT, design=design),
            })
    _write_csv("noise_metrics.csv", rows)

    figure, axis = plt.subplots(figsize=(3.4, 2.5))
    implemented = [r for r in rows if r["design"] == "implemented"]
    exact = [r for r in rows if r["design"] == "exact"]
    axis.loglog(bandwidths, [r["e3_h2"] for r in implemented], "o-", label=r"implemented: $H_2$")
    axis.loglog(bandwidths, [r["e3_h2"] for r in exact], "o--", label=r"exact poles: $H_2$")
    axis.loglog(bandwidths, [r["e3_linf_induced"] for r in implemented], "^-.", label=r"implemented: $\ell_\infty$")
    axis.loglog(bandwidths, [r["e3_linf_induced"] for r in exact], "^:", label=r"exact poles: $\ell_\infty$")
    axis.set_xlabel(r"observer bandwidth $\omega_o$ (rad/s)")
    axis.set_ylabel(r"unit-noise gain to $e_3$")
    axis.legend(frameon=False)
    _style(axis, log_grid=True)
    _save_figure(figure, "fig_norm_consistent_noise")
    return rows


def experiment_tuning() -> tuple[list[dict], list[dict]]:
    bandwidths = np.linspace(0.5, 20.0, 79)
    sigma_n = 0.01
    acceleration_per_wave_height = 0.02
    rows = []
    curves = {}
    for sea_state in range(1, 6):
        omega_p = 2.0 * np.pi / TP_TABLE[sea_state]
        sigma_d = acceleration_per_wave_height * HS_TABLE[sea_state]
        values = [tuning_criterion(o, omega_p, sigma_d, sigma_n, DT)["total_rms"] for o in bandwidths]
        optimum_index = int(np.argmin(values))
        optimum = float(bandwidths[optimum_index])
        curves[sea_state] = np.asarray(values)
        rows.append({
            "sea_state": sea_state,
            "Hs_m": HS_TABLE[sea_state],
            "Tp_s": TP_TABLE[sea_state],
            "omega_p_rad_s": omega_p,
            "assumed_sigma_d_m_s2": sigma_d,
            "assumed_sigma_n_m": sigma_n,
            "selected_omega_o_rad_s": optimum,
            "minimum_total_e3_rms_m_s2": values[optimum_index],
            "boundary_solution": optimum_index in (0, len(bandwidths) - 1),
        })
    _write_csv("jonswap_case_tuning.csv", rows)

    sensitivity_rows = []
    sea_state = 3
    omega_p = 2.0 * np.pi / TP_TABLE[sea_state]
    sigma_d = acceleration_per_wave_height * HS_TABLE[sea_state]
    for noise_sigma in (0.002, 0.005, 0.01, 0.02):
        values = [tuning_criterion(o, omega_p, sigma_d, noise_sigma, DT)["total_rms"] for o in bandwidths]
        optimum_index = int(np.argmin(values))
        sensitivity_rows.append({
            "sea_state": sea_state,
            "sigma_n_m": noise_sigma,
            "selected_omega_o_rad_s": float(bandwidths[optimum_index]),
            "minimum_total_e3_rms_m_s2": values[optimum_index],
            "boundary_solution": optimum_index in (0, len(bandwidths) - 1),
        })
    _write_csv("tuning_sensitivity.csv", sensitivity_rows)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for index, sea_state in enumerate((1, 3, 5)):
        axes[0].semilogy(bandwidths, curves[sea_state], color=COLORS[index], label=f"SS{sea_state}")
    axes[0].set_xlabel(r"observer bandwidth $\omega_o$ (rad/s)")
    axes[0].set_ylabel(r"predicted RMS of $e_3$ (m/s$^2$)")
    axes[0].legend(frameon=False)
    _style(axes[0])
    axes[1].semilogx([r["sigma_n_m"] for r in sensitivity_rows], [r["selected_omega_o_rad_s"] for r in sensitivity_rows], "o-", color=COLORS[1])
    axes[1].set_xlabel(r"measurement-noise standard deviation $\sigma_n$ (m)")
    axes[1].set_ylabel(r"selected $\omega_o$ (rad/s), SS3")
    _style(axes[1])
    _save_figure(figure, "fig_jonswap_case_tuning")
    return rows, sensitivity_rows


def write_summary(poles, sampling, band_share, noise, tuning, sensitivity) -> None:
    payload = {
        "model_scope": "discrete ESO audit with an illustrative JONSWAP disturbance-spectrum proxy",
        "sampling_period_s": DT,
        "physical_band_share": band_share,
        "assumptions": {
            "measurement_noise": "independent zero-mean samples; reported with H2/RMS metric",
            "JONSWAP_case_sigma_d": "0.02*Hs m/s^2 (explicit illustrative scale)",
            "JONSWAP_case_sigma_n": "0.01 m except sensitivity analysis",
            "full_loop_nonlinear_validation_claim": False,
        },
        "pole_audit": poles,
        "spectral_sampling": sampling,
        "noise_metrics": noise,
        "jonswap_case_tuning": tuning,
        "tuning_sensitivity": sensitivity,
    }
    (DATA / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    n32 = next(row for row in sampling if row["N"] == 32)
    lines = [
        "# Results — revised reproducible analysis", "",
        "All values in this file are generated by `validate_theory.py`.", "",
        "## Pole audit", "",
        "The implemented gain polynomial differs from the target by `(1-rho)^3 lambda`; the corrected gain vector places the triple pole.", "",
        "## Corrected JONSWAP sampling", "",
        f"- Physical band share over [0.6 omega_p, 2.4 omega_p]: {100*band_share:.3f}%.",
        f"- N=32 mean signed relative error: {n32['mean_signed_relative_error']:.4f}.",
        f"- N=32 standard deviation: {n32['standard_deviation_relative_error']:.4f}.", "",
        "## Norm-consistent noise metrics", "",
        "The H2/RMS factor, impulse peak, and bounded-noise induced gain are reported separately in `generated_data/noise_metrics.csv`.", "",
        "## JONSWAP case study", "",
        "The tuning example is explicitly illustrative: the JONSWAP density is used as a normalized disturbance-spectrum proxy with sigma_d=0.02*Hs m/s^2 and independent measurement noise. It is not presented as a full nonlinear closed-loop or hydrodynamic validation.", "",
        "| SS | selected omega_o (rad/s) | boundary solution |", "|---:|---:|:---:|",
    ]
    lines.extend(f"| {r['sea_state']} | {r['selected_omega_o_rad_s']:.2f} | {r['boundary_solution']} |" for r in tuning)
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "RESULTS_raw.txt").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    poles = experiment_poles()
    sampling, share = experiment_sampling()
    noise = experiment_noise()
    tuning, sensitivity = experiment_tuning()
    write_summary(poles, sampling, share, noise, tuning, sensitivity)
    from redraw_advanced_figures import main as redraw_advanced_figures

    redraw_advanced_figures()
    print(f"physical JONSWAP band share: {100*share:.3f}%")
    print(f"generated figures: {FIGURES}")
    print(f"generated data: {DATA}")


if __name__ == "__main__":
    main()
