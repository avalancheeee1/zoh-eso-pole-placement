# -*- coding: utf-8 -*-
"""Robustness evidence added for the first revision.

Addresses reviewer requests:
  R1.4  the true additive disturbance entering the plant is a combination of
        all three estimation-error components, not e3 alone;
  R1.6  comparison at matched spectral radius, so that the benefit of the gain
        correction is separated from a tuning mismatch;
  R1.7  a normalized argument (exact similarity transform) showing that the
        audit depends on omega_o*Ts only, plus a multi-Ts numerical check;
  R1.8  a closed-loop case with simultaneous measurement noise and disturbance;
  R3.4  the rebound under several controller bandwidths omega_c;
  R3.5  a high-precision location of the spectral-radius minimum;
  R3.2  symbolic verification of all four Jury conditions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redraw_advanced_figures import (
    GREY,
    INK,
    NAVY,
    ORANGE,
    PURPLE,
    TEAL,
    panel_label,
    save_figure,
    style_axis,
)
from analysis_core import (
    DT,
    error_matrix_from_gains,
    exact_zoh_poleplacement_gains,
    implemented_gains,
    noise_metrics,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_data"
OUT.mkdir(exist_ok=True)
B0 = 1.0
STEP_TIME = 2.0
D0 = 1.0


def gains(omega_o: float, design: str, dt: float = DT) -> np.ndarray:
    if design == "implemented":
        return implemented_gains(omega_o, dt)
    return exact_zoh_poleplacement_gains(omega_o, dt)


def spectral_radius(omega_o: float, design: str, dt: float = DT) -> float:
    matrix = error_matrix_from_gains(gains(omega_o, design, dt), dt)
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


# --------------------------------------------------------------------------
# R1.7 -- exact similarity transform: A depends on omega_o*Ts only
# --------------------------------------------------------------------------
def normalized_matrix(omega_o: float, design: str, dt: float = DT) -> np.ndarray:
    """A_norm = D^{-1} A D with D = diag(1, 1/Ts, 1/Ts^2).

    A_norm depends on a = 1 - exp(-omega_o*Ts) only, so A and A_norm have the
    same spectrum for every sampling period at fixed omega_o*Ts.
    """
    l1, l2, l3 = gains(omega_o, design, dt)
    return np.array(
        [
            [1.0 - l1, 1.0, 0.5],
            [-dt * l2, 1.0, 1.0],
            [-dt**2 * l3, 0.0, 1.0],
        ]
    )


def check_ts_invariance() -> dict:
    """Spectral radius must depend on omega_o*Ts only, across sampling periods."""
    rows = []
    for omega_o_ts in (0.2, 1.0, 2.0, 5.0):
        radii = {}
        for dt in (0.01, 0.02, 0.05, 0.1, 0.2):
            omega_o = omega_o_ts / dt
            radii[f"{dt:g}"] = spectral_radius(omega_o, "implemented", dt)
        rows.append(
            {
                "omega_o_Ts": omega_o_ts,
                "radii": radii,
                "spread": max(radii.values()) - min(radii.values()),
            }
        )
    # similarity residual at one working point
    dt = DT
    omega_o = 2.0
    matrix = error_matrix_from_gains(gains(omega_o, "implemented", dt), dt)
    scale = np.diag([1.0, 1.0 / dt, 1.0 / dt**2])
    rebuilt = scale @ normalized_matrix(omega_o, "implemented", dt) @ np.linalg.inv(scale)
    return {
        "rows": rows,
        "max_spread": max(r["spread"] for r in rows),
        "similarity_residual": float(np.max(np.abs(matrix - rebuilt))),
    }


# --------------------------------------------------------------------------
# R3.5 -- high-precision spectral-radius profile
# --------------------------------------------------------------------------
def spectral_radius_profile(x, design: str = "implemented"):
    """Spectral radius as a function of the normalized bandwidth x = omega_o*Ts.

    Uses the Ts-free characteristic polynomial in mu = lambda - 1:
    mu^3 + l1 mu^2 + (Ts*l2 + Ts^2*l3/2) mu + Ts^2*l3, whose coefficients are
    Ts-independent after the similarity transform.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    a = 1.0 - np.exp(-x)
    if design == "implemented":
        c2, c1, c0 = 3.0 * a, 3.0 * a**2 + a**3, 2.0 * a**3
    else:
        c2, c1, c0 = 3.0 * a, a**2 * (3.0 - 0.5 * a) + 0.5 * a**3, a**3
    radii = np.empty_like(x)
    for index in range(x.size):
        roots = np.roots([1.0, c2[index], c1[index], c0[index]]) + 1.0
        radii[index] = np.max(np.abs(roots))
    return radii


def refine_minimum(x_lo: float = 0.1, x_hi: float = 4.0) -> dict:
    """Golden-section refinement of the spectral-radius minimum."""
    def sr(x: float) -> float:
        a = 1.0 - np.exp(-x)
        coeffs = [1.0, 3.0 * a, 3.0 * a**2 + a**3, 2.0 * a**3]
        return float(np.max(np.abs(np.roots(coeffs) + 1.0)))

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = x_lo, x_hi
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = sr(c), sr(d)
    for _ in range(300):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = sr(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = sr(d)
        if hi - lo < 1e-14:
            break
    x_star = 0.5 * (lo + hi)
    return {
        "x_star": x_star,
        "min_spectral_radius": sr(x_star),
        "bracket_width": hi - lo,
        "radius_at_5": sr(5.0),
        "radius_at_10": sr(10.0),
        "radius_at_0.01": sr(0.01),
    }


# --------------------------------------------------------------------------
# R1.6 -- matched spectral radius
# --------------------------------------------------------------------------
def matched_spectral_radius_rows(omega_o_list) -> list[dict]:
    rows = []
    for omega_o in omega_o_list:
        sr_imp = spectral_radius(omega_o, "implemented")
        # the exact design places all poles at rho = exp(-omega_o' Ts), so its
        # spectral radius is exp(-omega_o' Ts); match by inverting that map.
        omega_o_matched = -np.log(sr_imp) / DT
        m_imp = noise_metrics(omega_o, DT, design="implemented")
        m_exact = noise_metrics(omega_o_matched, DT, design="exact")
        rows.append(
            {
                "omega_o": float(omega_o),
                "sr_implemented": sr_imp,
                "omega_o_matched": float(omega_o_matched),
                "h2_implemented": m_imp["e3_h2"],
                "h2_exact_matched": m_exact["e3_h2"],
                "linf_implemented": m_imp["e3_linf_induced"],
                "linf_exact_matched": m_exact["e3_linf_induced"],
                "h2_ratio": m_imp["e3_h2"] / m_exact["e3_h2"],
                "linf_ratio": m_imp["e3_linf_induced"] / m_exact["e3_linf_induced"],
            }
        )
    return rows


def equivalent_noise_bandwidth_rows(omega_o_list) -> list[dict]:
    """Complementary view: the bandwidth the correction effectively recovers.

    For each audited bandwidth, find the omega_o at which the corrected design
    has the same H2 factor as the audited design at that bandwidth.
    """
    grid = np.linspace(0.05, 300.0, 12000)
    h2_grid = np.array([noise_metrics(w, DT, design="exact")["e3_h2"] for w in grid])
    ceiling = float(h2_grid.max())
    rows = []
    for omega_o in omega_o_list:
        target = noise_metrics(omega_o, DT, design="implemented")["e3_h2"]
        # The corrected design's H2 factor saturates: its gains stay bounded as
        # omega_o -> infinity while all three poles tend to zero.
        exceeds = bool(target >= ceiling)
        omega_eq = None if exceeds else float(np.interp(target, h2_grid, grid))
        rows.append(
            {
                "omega_o": float(omega_o),
                "h2_implemented": target,
                "omega_o_equivalent_exact": omega_eq,
                # how much more bandwidth the corrected design supports at equal noise
                "bandwidth_ratio": None if exceeds else omega_eq / float(omega_o),
                "exceeds_corrected_ceiling": exceeds,
            }
        )
    return rows, ceiling


# --------------------------------------------------------------------------
# R1.4 / R1.8 -- instrumented closed loop
# --------------------------------------------------------------------------
def simulate_full(omega_o, design, *, t_end=8.0, step_time=STEP_TIME, d0=D0,
                  sigma_n=0.0, omega_c=3.0, zeta=1.0, seed=0, dt=DT):
    """Closed loop returning the estimation-error components and the true
    additive disturbance w = kp*e1 + kd*e2 + e3 that enters the plant."""
    rng = np.random.default_rng(seed)
    kp = omega_c**2
    kd = 2.0 * zeta * omega_c
    l1, l2, l3 = gains(omega_o, design, dt)

    F = np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    B = np.array([B0 * 0.5 * dt**2, B0 * dt, 0.0])
    H = np.array([1.0, 0.0, 0.0])
    L = np.array([l1, l2, l3])

    A2 = np.array([[1.0, dt], [0.0, 1.0]])
    Bu = np.array([B0 * 0.5 * dt**2, B0 * dt])
    Bd = np.array([0.5 * dt**2, dt])

    n_steps = int(round(t_end / dt))
    t = np.arange(n_steps) * dt
    x = np.zeros(2)
    z = np.zeros(3)
    out = {k: np.zeros(n_steps) for k in
           ("y", "e1", "e2", "e3", "w", "u", "d")}
    for k in range(n_steps):
        d = d0 if t[k] >= step_time else 0.0
        y = x[0] + rng.normal(0.0, sigma_n)
        u0 = kp * (0.0 - z[0]) - kd * z[1]
        u = (u0 - z[2]) / B0
        x = A2 @ x + Bu * u + Bd * d
        z = F @ z + B * u + L * (y - H @ z)
        e1, e2, e3 = x[0] - z[0], x[1] - z[1], d - z[2]
        out["y"][k] = x[0]
        out["e1"][k] = e1
        out["e2"][k] = e2
        out["e3"][k] = e3
        # plant: x_ddot = -kp*x1 - kd*x2 + (kp*e1 + kd*e2 + e3)
        out["w"][k] = kp * e1 + kd * e2 + e3
        out["u"][k] = u
        out["d"][k] = d
    return t, out


def post_step(t):
    return int(np.searchsorted(t, STEP_TIME))


def closed_loop_table(bandwidths, omega_c_list, sigma_n) -> dict:
    result = {"by_omega_c": {}, "noisy": None}
    for omega_c in omega_c_list:
        rows = []
        for w in bandwidths:
            row = {"omega_o": float(w)}
            for design in ("implemented", "exact"):
                t, out = simulate_full(w, design, omega_c=omega_c)
                j = post_step(t)
                row[f"rms_e3_{design}"] = float(np.sqrt(np.mean(out["e3"][j:] ** 2)))
                row[f"rms_w_{design}"] = float(np.sqrt(np.mean(out["w"][j:] ** 2)))
            rows.append(row)
        result["by_omega_c"][f"{omega_c:g}"] = rows

    # simultaneous noise and disturbance (R1.8)
    noisy = []
    for w in (5.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0):
        row = {"omega_o": float(w)}
        for design in ("implemented", "exact"):
            acc_e3, acc_w = [], []
            for seed in range(8):
                t, out = simulate_full(w, design, t_end=8.0, sigma_n=sigma_n, seed=seed)
                j = post_step(t)
                acc_e3.append(np.sqrt(np.mean(out["e3"][j:] ** 2)))
                acc_w.append(np.sqrt(np.mean(out["w"][j:] ** 2)))
            row[f"rms_e3_{design}"] = float(np.mean(acc_e3))
            row[f"rms_w_{design}"] = float(np.mean(acc_w))
        noisy.append(row)
    result["noisy"] = noisy
    return result


# --------------------------------------------------------------------------
# R3.2 -- all four Jury conditions, verified symbolically
# --------------------------------------------------------------------------
def jury_check():
    import sympy as sp

    rho = sp.symbols("rho", positive=True)
    lam = sp.symbols("lambda")
    a = 1 - rho
    p = sp.expand(lam**3 - 3 * rho * lam**2 + (1 - 3 * rho + 6 * rho**2 - rho**3) * lam - rho**3)
    a0 = sp.Poly(p, lam).coeff_monomial(1)
    a1 = sp.Poly(p, lam).coeff_monomial(lam)
    a2 = sp.Poly(p, lam).coeff_monomial(lam**2)
    cond1 = sp.factor(sp.simplify(p.subs(lam, 1)))
    cond2 = sp.factor(sp.simplify(-p.subs(lam, -1)))
    cond3 = sp.factor(sp.simplify(1 - a0**2))
    b0 = sp.factor(sp.simplify(1 - a0**2))
    b2 = sp.factor(sp.simplify(sp.simplify(a0 * a2 - a1)))
    cond4 = sp.factor(sp.simplify(b0 - b2))
    cond4b = sp.factor(sp.simplify(b0 + b2))
    c0 = sp.factor(sp.simplify(b0**2 - b2**2))
    return {
        "cond1_p(1)": sp.srepr(cond1),
        "cond1_pretty": str(cond1),
        "cond2_minus_p(-1)": str(cond2),
        "cond3_1_minus_a0_squared": str(cond3),
        "b0": str(b0),
        "b2": str(b2),
        "cond4_b0_minus_b2": str(cond4),
        "cond4_b0_plus_b2": str(cond4b),
        "c0": str(c0),
        "c0_factored": str(sp.factor(c0)),
    }


def draw_figure(report: dict) -> None:
    figure, axes = plt.subplots(
        2, 3, figsize=(7.15, 4.5),
        gridspec_kw={"wspace": 0.42, "hspace": 0.62},
    )

    # (a) R1.7 -- sampling-period invariance
    ax = axes[0, 0]
    x_curve = np.linspace(0.02, 3.0, 240)
    ax.plot(x_curve, spectral_radius_profile(x_curve), color=NAVY, lw=1.3,
            label="implemented", zorder=2)
    ax.plot(x_curve, np.exp(-x_curve), color=GREY, ls="--", lw=1.1,
            label=r"target $\rho$", zorder=2)
    for dt, marker, colour in [(0.01, "o", ORANGE), (0.02, "s", TEAL),
                               (0.05, "^", PURPLE), (0.2, "d", "#B8860B")]:
        xs = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
        ys = np.array([spectral_radius(v / dt, "implemented", dt) for v in xs])
        ax.plot(xs, ys, marker, ms=3.6, mfc="white", mec=colour, mew=0.9,
                ls="none", zorder=3)
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    dt_handles = [
        Line2D([], [], marker=m, ls="none", mfc="white", mec=c, mew=0.9, ms=3.8,
               label=rf"$T_s$ = {d:g} s")
        for d, m, c in [(0.01, "o", ORANGE), (0.02, "s", TEAL),
                        (0.05, "^", PURPLE), (0.2, "d", "#B8860B")]
    ]
    ax.legend(handles + dt_handles, labels + [h.get_label() for h in dt_handles],
              frameon=False, loc="lower left", fontsize=5.9, handlelength=1.5,
              labelspacing=0.28, borderpad=0.1)
    ax.set(xlabel=r"normalized bandwidth $\omega_o T_s$",
           ylabel="spectral radius", xlim=(0, 3.0), ylim=(0, 1.05))
    style_axis(ax, grid="y")
    panel_label(ax, "a")

    # (b) R1.6 -- matched spectral radius
    ax = axes[0, 1]
    rows = report["matched_spectral_radius"]
    w = np.array([r["omega_o"] for r in rows])
    ax.semilogy(w, [r["h2_ratio"] for r in rows], "o-", color=ORANGE,
                label=r"$H_2$/RMS factor")
    ax.semilogy(w, [r["linf_ratio"] for r in rows], "s--", color=NAVY,
                mfc="white", label=r"induced $\ell_\infty$ gain")
    ax.set(xlabel=r"observer bandwidth $\omega_o$ (rad s$^{-1}$)",
           ylabel="implemented / exact ratio")
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    style_axis(ax, grid="y")
    panel_label(ax, "b")

    # (c) R3.5 -- fine spectral-radius profile with located minimum
    ax = axes[0, 2]
    x_fine = np.linspace(1e-3, 12.0, 4000)
    ax.plot(x_fine, spectral_radius_profile(x_fine), color=NAVY, lw=1.3)
    mm = report["radius_minimum"]
    ax.plot(mm["x_star"], mm["min_spectral_radius"], "o", ms=4.6, color=ORANGE,
            mec="white", mew=0.8, zorder=4)
    ax.annotate(rf"min {mm['min_spectral_radius']:.4f}" "\n"
                rf"at $\omega_o T_s$ = {mm['x_star']:.3f}",
                xy=(mm["x_star"], mm["min_spectral_radius"]),
                xytext=(2.6, 0.86), fontsize=6.0, color=GREY,
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8})
    ax.axhline(1.0, color=GREY, ls=":", lw=0.8)
    ax.set(xlabel=r"normalized bandwidth $\omega_o T_s$",
           ylabel="implemented spectral radius", xlim=(0, 12), ylim=(0.7, 1.02))
    style_axis(ax, grid="y")
    panel_label(ax, "c")

    # (d) R1.4 + R3.4 -- true injected disturbance across controller bandwidths
    ax = axes[1, 0]
    for (omega_c, colour) in [("1", PURPLE), ("3", ORANGE), ("10", TEAL)]:
        rows_c = report["closed_loop"]["by_omega_c"][omega_c]
        ww = np.array([r["omega_o"] for r in rows_c])
        ax.semilogx(ww, [r["rms_w_implemented"] for r in rows_c], "-",
                    color=colour, label=rf"impl., $\omega_c$ = {omega_c}")
        ax.semilogx(ww, [r["rms_w_exact"] for r in rows_c], "--",
                    color=colour, alpha=0.65)
    peak_d = max(
        max(r["rms_w_implemented"] for r in report["closed_loop"]["by_omega_c"][c])
        for c in ("1", "3", "10")
    )
    ax.set(xlabel=r"$\omega_o$ (rad s$^{-1}$)",
           ylabel=r"RMS of injected $w$ (m s$^{-2}$)",
           ylim=(0.0, peak_d * 1.45))
    ax.legend(frameon=False, loc="upper left", fontsize=5.9, handlelength=2.0,
              labelspacing=0.3)
    style_axis(ax, grid="both")
    panel_label(ax, "d")

    # (e) R1.8 -- simultaneous measurement noise and disturbance
    ax = axes[1, 1]
    rows_n = report["closed_loop"]["noisy"]
    wn = np.array([r["omega_o"] for r in rows_n])
    ax.loglog(wn, [r["rms_w_implemented"] for r in rows_n], "o-", color=ORANGE,
              label="implemented")
    ax.loglog(wn, [r["rms_w_exact"] for r in rows_n], "s--", color=NAVY,
              mfc="white", label="exact poles")
    ax.set(xlabel=r"$\omega_o$ (rad s$^{-1}$)",
           ylabel=r"RMS of injected $w$ (m s$^{-2}$)")
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    style_axis(ax, grid="both")
    panel_label(ax, "e")

    # (f) R1.4 -- the three estimation-error contributions to w
    ax = axes[1, 2]
    t, out = simulate_full(100.0, "implemented", t_end=3.6, omega_c=3.0)
    kp, kd = 9.0, 6.0
    ax.plot(t, np.abs(kp * out["e1"]), color=NAVY, lw=1.1,
            label=r"$k_p e_1$")
    ax.plot(t, np.abs(kd * out["e2"]), color=TEAL, lw=1.1,
            label=r"$k_d e_2$")
    ax.plot(t, np.abs(out["e3"]), color=ORANGE, lw=1.1, label=r"$e_3$")
    ax.set(xlabel="time (s)", ylabel=r"contribution to $w$ (m s$^{-2}$)",
           xlim=(2.0, 3.6), yscale="log", ylim=(1e-4, 4e0))
    ax.legend(frameon=False, loc="lower left", handlelength=1.8, ncols=3,
              columnspacing=1.0, handletextpad=0.5)
    style_axis(ax, grid="y")
    panel_label(ax, "f")

    save_figure(figure, "fig_rev_robustness")
    print("saved figures/fig_rev_robustness.*")


def write_closed_loop_csv(report: dict) -> None:
    """Export the RMS perturbation entering the plant, in machine-readable form.

    One tidy row per (controller bandwidth, observer bandwidth, noise level), so
    the table reported in the manuscript is a direct projection of this file.
    """
    import csv

    rows = []
    for omega_c, block in report["closed_loop"]["by_omega_c"].items():
        for r in block:
            rows.append({
                "omega_c": float(omega_c),
                "omega_o": r["omega_o"],
                "sigma_n": 0.0,
                "rms_perturbation_implemented": r["rms_w_implemented"],
                "rms_perturbation_exact": r["rms_w_exact"],
            })
    for r in report["closed_loop"]["noisy"]:
        rows.append({
            "omega_c": 3.0,
            "omega_o": r["omega_o"],
            "sigma_n": 0.01,
            "rms_perturbation_implemented": r["rms_w_implemented"],
            "rms_perturbation_exact": r["rms_w_exact"],
        })
    target = OUT / "closed_loop_robustness.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {target}")


def main() -> None:
    report: dict = {}

    print("== R1.7  Ts invariance ==")
    inv = check_ts_invariance()
    report["ts_invariance"] = inv
    for r in inv["rows"]:
        print(f"  omega_o*Ts={r['omega_o_Ts']:<5} spread={r['spread']:.3e}  {r['radii']}")
    print(f"  similarity residual = {inv['similarity_residual']:.3e}")

    print("\n== R3.5  spectral-radius minimum ==")
    m = refine_minimum()
    report["radius_minimum"] = m
    for k, v in m.items():
        print(f"  {k} = {v!r}")

    print("\n== R1.6  matched spectral radius ==")
    rows = matched_spectral_radius_rows([10.0, 20.0, 30.0, 60.0, 100.0])
    report["matched_spectral_radius"] = rows
    print("  w_o    SR_imp   w_o'    H2 imp  H2 exact  ratio   linf imp linf exact ratio")
    for r in rows:
        print(f"  {r['omega_o']:6.1f} {r['sr_implemented']:.4f} {r['omega_o_matched']:7.2f}"
              f" {r['h2_implemented']:8.3g} {r['h2_exact_matched']:9.3g} {r['h2_ratio']:6.2f}"
              f" {r['linf_implemented']:9.4g} {r['linf_exact_matched']:9.4g} {r['linf_ratio']:6.2f}")

    print("\n== R1.4/R1.8/R3.4  closed loop ==")
    bw = [5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    cl = closed_loop_table(bw, [1.0, 3.0, 10.0], sigma_n=0.01)
    report["closed_loop"] = cl
    for omega_c, rows_ in cl["by_omega_c"].items():
        print(f"  omega_c={omega_c}")
        for r in rows_:
            print(f"    w_o={r['omega_o']:6.1f}  RMS e3 {r['rms_e3_implemented']:.3e}/{r['rms_e3_exact']:.3e}"
                  f"   RMS w {r['rms_w_implemented']:.3e}/{r['rms_w_exact']:.3e}")
    print("  noisy case (sigma_n=0.01 m, 8 seeds)")
    for r in cl["noisy"]:
        print(f"    w_o={r['omega_o']:6.1f}  RMS e3 {r['rms_e3_implemented']:.3e}/{r['rms_e3_exact']:.3e}"
              f"   RMS w {r['rms_w_implemented']:.3e}/{r['rms_w_exact']:.3e}")

    print("\n== R1.6b  equivalent noise bandwidth ==")
    eq, ceiling = equivalent_noise_bandwidth_rows([10.0, 30.0, 60.0, 100.0])
    report["equivalent_noise_bandwidth"] = eq
    report["exact_h2_ceiling"] = ceiling
    print(f"  corrected-design H2 ceiling = {ceiling:.2f}")
    for r in eq:
        if r["exceeds_corrected_ceiling"]:
            print(f"  w_o={r['omega_o']:6.1f}  H2={r['h2_implemented']:9.4g}"
                  f"  above the corrected design's entire range")
        else:
            print(f"  w_o={r['omega_o']:6.1f}  H2={r['h2_implemented']:9.4g}"
                  f"  equals corrected design at w_o={r['omega_o_equivalent_exact']:6.3f}"
                  f"  ratio={r['bandwidth_ratio']:5.2f}x")

    print("\n== R3.2  Jury conditions ==")
    jury = jury_check()
    report["jury"] = jury
    for k, v in jury.items():
        print(f"  {k} = {v}")

    draw_figure(report)

    (OUT / "rev_robustness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'rev_robustness.json'}")
    write_closed_loop_csv(report)


if __name__ == "__main__":
    main()
