"""Create publication-grade figures from the manuscript's verified data.

The script does not alter numerical results. It reads the generated CSV files
and calls the same verified tuning function used by ``validate_theory.py``.
Each figure is exported as PDF, SVG, PNG, TIFF, and a grayscale PNG preview.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

from analysis_core import DT, HS_TABLE, TP_TABLE, tuning_criterion


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "generated_data"
FIGURES = ROOT / "figures"
LEGACY = ROOT / "figures_legacy_20260818"

NAVY = "#1F4E79"
ORANGE = "#D55E00"
TEAL = "#1B9E77"
PURPLE = "#7B3294"
GREY = "#5F6368"
LIGHT_GREY = "#D9DEE3"
INK = "#202124"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.1,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.55,
        "lines.markersize": 4.3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis, *, grid: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(colors=INK, direction="out", length=3.2)
    if grid == "y":
        axis.grid(axis="y", which="major", color=LIGHT_GREY, lw=0.55, alpha=0.72)
    elif grid == "both":
        axis.grid(which="major", color=LIGHT_GREY, lw=0.5, alpha=0.65)
    axis.set_axisbelow(True)


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.pdf")
    figure.savefig(FIGURES / f"{stem}.svg")
    figure.savefig(FIGURES / f"{stem}.png", dpi=400)
    figure.savefig(FIGURES / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)

    image = plt.imread(FIGURES / f"{stem}.png")
    rgb = image[..., :3]
    grey = np.dot(rgb, np.array([0.2126, 0.7152, 0.0722]))
    plt.imsave(FIGURES / f"{stem}_grayscale.png", grey, cmap="gray", vmin=0, vmax=1)


def figure_pole_audit() -> None:
    rows = read_csv("pole_audit.csv")
    x = np.array([float(row["omega_o_Ts"]) for row in rows])
    target = np.array([float(row["rho"]) for row in rows])
    implemented = np.array([float(row["implemented_spectral_radius"]) for row in rows])
    exact = np.array([float(row["exact_spectral_radius"]) for row in rows])
    error = np.array([float(row["polynomial_coefficient_error"]) for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.62), gridspec_kw={"wspace": 0.34})
    axis = axes[0]
    axis.axvspan(0.08, 0.2, color=TEAL, alpha=0.08, lw=0)
    axis.fill_between(x, target, implemented, color=ORANGE, alpha=0.12, lw=0)
    axis.semilogx(x, implemented, "o-", color=ORANGE, label="implemented")
    axis.semilogx(x, exact, "s-", color=NAVY, label="exact placement")
    axis.semilogx(x, target, "--", color=INK, lw=1.15, label=r"target $e^{-\omega_oT_s}$")
    axis.axhline(1.0, color=GREY, ls=":", lw=0.9)
    axis.annotate(
        "implemented poles rebound",
        xy=(5.0, implemented[-2]),
        xytext=(1.05, 0.57),
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8},
        color=GREY,
        fontsize=7,
    )
    axis.text(0.105, 0.32, "small-step\nregime", color=TEAL, fontsize=6.8)
    axis.set(xlabel=r"normalized bandwidth  $\omega_oT_s$", ylabel="spectral radius", ylim=(-0.02, 1.055))
    axis.legend(frameon=False, loc="lower left", handlelength=2.6)
    style_axis(axis, grid="y")
    panel_label(axis, "a")

    axis = axes[1]
    theory = (1.0 - np.exp(-x)) ** 3
    axis.loglog(x, error, "o", color=ORANGE, zorder=3, label="computed")
    axis.loglog(x, theory, "-", color=INK, lw=1.2, label=r"exact defect  $(1-\rho)^3$")
    axis.fill_between(x, 1e-4, error, color=ORANGE, alpha=0.10, lw=0)
    axis.annotate(
        r"25% at $\omega_oT_s=1$",
        xy=(1.0, error[3]),
        xytext=(0.19, 0.54),
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8},
        color=GREY,
        fontsize=7,
    )
    axis.set(xlabel=r"normalized bandwidth  $\omega_oT_s$", ylabel="characteristic-coefficient error", ylim=(5e-4, 1.35))
    axis.legend(frameon=False, loc="lower right", handlelength=2.4)
    style_axis(axis, grid="both")
    panel_label(axis, "b")
    save_figure(figure, "fig_pole_audit")


def figure_sampling() -> None:
    rows = read_csv("spectral_sampling.csv")
    n = np.array([float(row["N"]) for row in rows])
    bias = np.array([float(row["mean_signed_relative_error"]) for row in rows])
    mae = np.array([float(row["mean_absolute_relative_error"]) for row in rows])
    std = np.array([float(row["standard_deviation_relative_error"]) for row in rows])
    sem = np.array([float(row["standard_error_mean"]) for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), gridspec_kw={"wspace": 0.34})
    axis = axes[0]
    axis.axhspan(-0.01, 0.01, color=TEAL, alpha=0.09, lw=0)
    axis.axhline(0, color=INK, lw=0.9)
    axis.errorbar(n, bias, yerr=1.96 * sem, fmt="o-", color=ORANGE, capsize=2.2, elinewidth=0.9)
    axis.set_xscale("log", base=2)
    axis.set_xticks(n)
    axis.set_xticklabels([str(int(value)) for value in n], rotation=35)
    axis.set(xlabel="number of components  $N$", ylabel="signed relative error")
    axis.text(0.97, 0.08, "95% CI of the mean", transform=axis.transAxes, ha="right", color=GREY, fontsize=7)
    style_axis(axis, grid="y")
    panel_label(axis, "a")

    axis = axes[1]
    reference = std[0] * np.sqrt(n[0] / n)
    axis.loglog(n, std, "s-", color=NAVY, label="standard deviation")
    axis.loglog(n, mae, "o-", color=TEAL, label="mean absolute error")
    axis.loglog(n, reference, "--", color=INK, lw=1.05, label=r"$N^{-1/2}$ reference")
    axis.fill_between(n, 0.8 * reference, 1.2 * reference, color=GREY, alpha=0.08, lw=0)
    slope = np.polyfit(np.log(n), np.log(std), 1)[0]
    axis.text(0.97, 0.92, f"fitted slope = {slope:.2f}", transform=axis.transAxes, ha="right", va="top", fontsize=7, color=GREY)
    axis.set(xlabel="number of components  $N$", ylabel="relative dispersion")
    axis.legend(frameon=False, loc="lower left", handlelength=2.5)
    style_axis(axis, grid="both")
    panel_label(axis, "b")
    save_figure(figure, "fig_corrected_sampling")


def figure_noise() -> None:
    rows = read_csv("noise_metrics.csv")
    implemented = [row for row in rows if row["design"] == "implemented"]
    exact = [row for row in rows if row["design"] == "exact"]
    omega = np.array([float(row["omega_o"]) for row in implemented])
    h2_imp = np.array([float(row["e3_h2"]) for row in implemented])
    h2_exact = np.array([float(row["e3_h2"]) for row in exact])
    li_imp = np.array([float(row["e3_linf_induced"]) for row in implemented])
    li_exact = np.array([float(row["e3_linf_induced"]) for row in exact])

    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.62), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.34})
    axis = axes[0]
    axis.loglog(omega, h2_imp, "o-", color=NAVY)
    axis.loglog(omega, h2_exact, "o--", color=NAVY, markerfacecolor="white")
    axis.loglog(omega, li_imp, "^-", color=ORANGE)
    axis.loglog(omega, li_exact, "^--", color=ORANGE, markerfacecolor="white")
    metric_legend = axis.legend(
        handles=[
            Line2D([0], [0], color=NAVY, marker="o", label=r"$H_2$ / RMS"),
            Line2D([0], [0], color=ORANGE, marker="^", label=r"induced $\ell_\infty$"),
        ],
        frameon=False,
        loc="upper left",
    )
    axis.add_artist(metric_legend)
    axis.legend(
        handles=[
            Line2D([0], [0], color=GREY, ls="-", label="implemented"),
            Line2D([0], [0], color=GREY, ls="--", marker="o", markerfacecolor="white", label="exact poles"),
        ],
        frameon=False,
        loc="lower right",
    )
    axis.set(xlabel=r"observer bandwidth  $\omega_o$ (rad s$^{-1}$)", ylabel=r"unit-noise gain to $e_3$")
    style_axis(axis, grid="both")
    panel_label(axis, "a")

    axis = axes[1]
    h2_ratio = h2_imp / h2_exact
    li_ratio = li_imp / li_exact
    axis.semilogx(omega, h2_ratio, "o-", color=NAVY, label=r"$H_2$ / RMS")
    axis.semilogx(omega, li_ratio, "^-", color=ORANGE, label=r"induced $\ell_\infty$")
    axis.axhline(1.0, color=INK, lw=0.9)
    axis.fill_between(omega, 1, li_ratio, color=ORANGE, alpha=0.08, lw=0)
    axis.text(0.97, 0.10, "values > 1 favour\nexact pole placement", transform=axis.transAxes, ha="right", color=GREY, fontsize=7)
    axis.annotate(f"{li_ratio[-1]:.1f}×", xy=(omega[-1], li_ratio[-1]), xytext=(17, 4.15), arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8}, color=ORANGE, fontsize=7.3, fontweight="bold")
    axis.set(xlabel=r"observer bandwidth  $\omega_o$ (rad s$^{-1}$)", ylabel="implemented / exact gain", ylim=(0.85, 4.45))
    axis.legend(frameon=False, loc="upper left")
    style_axis(axis, grid="y")
    panel_label(axis, "b")
    save_figure(figure, "fig_norm_consistent_noise")


def figure_tuning() -> None:
    tuning_rows = read_csv("jonswap_case_tuning.csv")
    sensitivity_rows = read_csv("tuning_sensitivity.csv")
    bandwidths = np.linspace(0.5, 20.0, 79)
    sigma_n = 0.01
    curves: dict[int, np.ndarray] = {}
    for sea_state in (1, 3, 5):
        omega_p = 2.0 * np.pi / TP_TABLE[sea_state]
        sigma_d = 0.02 * HS_TABLE[sea_state]
        curves[sea_state] = np.array(
            [tuning_criterion(value, omega_p, sigma_d, sigma_n, DT)["total_rms"] for value in bandwidths]
        )

    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.46), gridspec_kw={"width_ratios": [1.35, 0.82, 1.0], "wspace": 0.43})
    axis = axes[0]
    for sea_state, color, line_style in zip((1, 3, 5), (NAVY, ORANGE, TEAL), ("-", "--", "-.")):
        values = curves[sea_state]
        index = int(np.argmin(values))
        axis.semilogy(bandwidths, values, color=color, ls=line_style, label=f"SS{sea_state}")
        axis.plot(bandwidths[index], values[index], "o", color=color, markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        if sea_state != 1:
            axis.text(bandwidths[index] + 0.35, values[index] * 0.87, f"{bandwidths[index]:.2g}", color=color, fontsize=6.7)
    axis.axvline(0.5, color=GREY, ls=":", lw=0.85)
    axis.set(xlabel=r"bandwidth  $\omega_o$ (rad s$^{-1}$)", ylabel=r"predicted RMS of $e_3$ (m s$^{-2}$)")
    axis.legend(frameon=False, loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.6)
    style_axis(axis, grid="y")
    panel_label(axis, "a")

    axis = axes[1]
    states = np.array([int(row["sea_state"]) for row in tuning_rows])
    selected = np.array([float(row["selected_omega_o_rad_s"]) for row in tuning_rows])
    boundary = np.array([row["boundary_solution"].lower() == "true" for row in tuning_rows])
    axis.plot(states, selected, "-", color=LIGHT_GREY, lw=1.2, zorder=1)
    axis.scatter(states[~boundary], selected[~boundary], s=28, color=TEAL, edgecolor="white", linewidth=0.7, zorder=3, label="interior")
    axis.scatter(states[boundary], selected[boundary], s=30, facecolor="white", edgecolor=ORANGE, linewidth=1.2, zorder=3, label="boundary")
    axis.set_xticks(states)
    axis.set(xlabel="sea state", ylabel=r"selected $\omega_o$ (rad s$^{-1}$)", ylim=(0.25, 2.82))
    axis.legend(frameon=False, loc="upper left", handletextpad=0.4)
    style_axis(axis, grid="y")
    panel_label(axis, "b")

    axis = axes[2]
    noise = np.array([float(row["sigma_n_m"]) for row in sensitivity_rows])
    selected_noise = np.array([float(row["selected_omega_o_rad_s"]) for row in sensitivity_rows])
    axis.semilogx(noise, selected_noise, "o-", color=PURPLE)
    axis.fill_between(noise, selected_noise, 1.5, color=PURPLE, alpha=0.08, lw=0)
    axis.annotate("higher noise\n→ lower bandwidth", xy=(0.02, selected_noise[-1]), xytext=(0.003, 2.15), arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8}, color=GREY, fontsize=7)
    axis.set(xlabel=r"noise s.d.  $\sigma_n$ (m)", ylabel=r"selected $\omega_o$ (rad s$^{-1}$)", ylim=(1.48, 3.48))
    style_axis(axis, grid="y")
    panel_label(axis, "c")
    save_figure(figure, "fig_jonswap_case_tuning")


def comparison_overview() -> None:
    """Before/after redesign sheet. Skipped when the legacy renders are absent.

    The legacy renders are an internal working artifact and are not part of this
    repository, so the overview is optional and never blocks a clean run.
    """
    if not LEGACY.is_dir():
        print(f"[skip] comparison_overview: {LEGACY.name}/ is not present")
        return
    stems = [
        "fig_pole_audit",
        "fig_corrected_sampling",
        "fig_norm_consistent_noise",
        "fig_jonswap_case_tuning",
    ]
    titles = ["Pole-placement audit", "Monte Carlo convergence", "Noise amplification", "JONSWAP-proxy tuning"]
    figure, axes = plt.subplots(4, 2, figsize=(11.7, 12.0), gridspec_kw={"hspace": 0.28, "wspace": 0.04})
    for row_index, (stem, title) in enumerate(zip(stems, titles)):
        old = plt.imread(LEGACY / f"{stem}.png")
        new = plt.imread(FIGURES / f"{stem}.png")
        for column, image in enumerate((old, new)):
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
        axes[row_index, 0].text(0.0, 1.04, title, transform=axes[row_index, 0].transAxes, fontsize=10, fontweight="bold", va="bottom")
    axes[0, 0].text(0.5, 1.18, "BEFORE", transform=axes[0, 0].transAxes, fontsize=13, fontweight="bold", color=GREY, ha="center")
    axes[0, 1].text(0.5, 1.18, "AFTER — NATURE-STYLE EVIDENCE DESIGN", transform=axes[0, 1].transAxes, fontsize=13, fontweight="bold", color=NAVY, ha="center")
    figure.savefig(FIGURES / "figure_redesign_before_after.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "figure_redesign_before_after.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_contracts() -> None:
    text = """# Figure contracts and visual specification

## Global specification

- Backend: Python/Matplotlib; data are unchanged from the verified CSV/analysis functions.
- Target: compact, Nature-inspired evidence figures suitable for a two-column journal.
- Colour: colour-vision-safe navy/orange/teal/purple; line style and marker shape provide redundant encoding.
- Typography: Arial/Helvetica fallback, 7.3–8.5 pt at final export size.
- Outputs: editable PDF/SVG, 400 dpi PNG, 600 dpi LZW TIFF, and grayscale preview.

## Figure contracts

1. **Pole audit** — Finding: the implemented gains increasingly miss the requested pole placement as normalized bandwidth grows. Evidence: target/exact/implemented spectral radii and the analytic coefficient defect. Visual test: the rebound and the 25% error at unity must be visible without reading the caption.
2. **Sampling convergence** — Finding: the corrected log-uniform estimator is unbiased within Monte Carlo uncertainty and its dispersion follows the expected square-root law. Evidence: signed-error 95% intervals plus standard deviation/MAE scaling. Visual test: zero crossing and the fitted slope must be readable.
3. **Noise amplification** — Finding: exact pole placement reduces both RMS and bounded-noise gains across the tested range. Evidence: absolute curves and implemented/exact ratios. Visual test: ratio remains above one and the high-bandwidth advantage is explicit.
4. **JONSWAP tuning** — Finding: the illustrative criterion has interior optima for higher sea states and selects lower bandwidth as measurement noise increases. Evidence: objective curves, all-state selections with boundary coding, and SS3 sensitivity. Visual test: boundary solutions cannot be mistaken for interior optima.
"""
    (ROOT / "FIGURE_CONTRACTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    figure_pole_audit()
    figure_sampling()
    figure_noise()
    figure_tuning()
    comparison_overview()
    write_contracts()
    print("Advanced figures written to", FIGURES)


if __name__ == "__main__":
    main()
