# -*- coding: utf-8 -*-
"""Closed-loop ADRC simulation: audited vs exact ZOH-pole-placement gains.

Demonstrates the paper's thesis end-to-end. The plant is the sampled
double-integrator channel x_ddot = b0 u + d; the third-order ESO estimates
[position, velocity, disturbance] and the ADRC law u = (u0 - z3)/b0 cancels
the estimated disturbance with a PD nominal feedback u0 = kp(r - z1) - kd z2.
At high observer bandwidth the audited gain vector (l3 = 2 a^3/Ts^2) places
its poles off-target, the observer spectral radius rebounds toward one, and
closed-loop disturbance rejection degrades relative to the exact triple-pole
design.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redraw_advanced_figures import (
    NAVY, ORANGE, TEAL, GREY, INK, style_axis, panel_label, save_figure,
)
from analysis_core import DT, implemented_gains, exact_zoh_poleplacement_gains

TS = DT          # 0.05 s
B0 = 1.0         # input gain (cancels in the ADRC law)
STEP_TIME = 2.0  # s
D0 = 1.0         # step disturbance magnitude (m/s^2)


def observer_gains(omega_o: float, design: str) -> np.ndarray:
    if design == "implemented":
        return implemented_gains(omega_o, TS)
    return exact_zoh_poleplacement_gains(omega_o, TS)


def simulate(omega_o, design, *, t_end=8.0, step_time=STEP_TIME, d0=D0,
             sigma_n=0.0, omega_c=3.0, zeta=1.0, seed=0):
    """Return (t, y, e3, u, d) for a closed-loop step-disturbance response."""
    rng = np.random.default_rng(seed)
    kp = omega_c**2
    kd = 2.0 * zeta * omega_c
    l1, l2, l3 = observer_gains(omega_o, design)

    # triple-integrator observer prediction
    F = np.array([[1.0, TS, 0.5 * TS**2],
                  [0.0, 1.0, TS],
                  [0.0, 0.0, 1.0]])
    B = np.array([B0 * 0.5 * TS**2, B0 * TS, 0.0])
    H = np.array([1.0, 0.0, 0.0])
    L = np.array([l1, l2, l3])

    # double-integrator plant (ZOH, disturbance enters as acceleration)
    A2 = np.array([[1.0, TS], [0.0, 1.0]])
    Bu = np.array([B0 * 0.5 * TS**2, B0 * TS])
    Bd = np.array([0.5 * TS**2, TS])

    N = int(round(t_end / TS))
    t = np.arange(N) * TS
    x = np.zeros(2)   # [position, velocity]
    z = np.zeros(3)   # [pos_est, vel_est, dist_est]
    u = 0.0
    y_out = np.zeros(N)
    e3_out = np.zeros(N)
    u_out = np.zeros(N)
    d_out = np.zeros(N)
    for k in range(N):
        d = d0 if t[k] >= step_time else 0.0
        n = rng.normal(0.0, sigma_n)
        y = x[0] + n
        # control from the current estimate
        u0 = kp * (0.0 - z[0]) - kd * z[1]
        u = (u0 - z[2]) / B0
        # plant advances with this control
        x = A2 @ x + Bu * u + Bd * d
        # observer updates with the same control and the current measurement
        z = F @ z + B * u + L * (y - H @ z)
        y_out[k] = x[0]
        e3_out[k] = d - z[2]
        u_out[k] = u
        d_out[k] = d
    return t, y_out, e3_out, u_out, d_out


def post_step_slice(t):
    return int(np.searchsorted(t, STEP_TIME))


def metric_peak_y(omega_o, design, **kw):
    t, y, e3, u, d = simulate(omega_o, design, **kw)
    j = post_step_slice(t)
    return float(np.max(np.abs(y[j:])))


def metric_rms_e3(omega_o, design, **kw):
    t, y, e3, u, d = simulate(omega_o, design, **kw)
    j = post_step_slice(t)
    return float(np.sqrt(np.mean(e3[j:] ** 2)))


def main():
    # ---- sweep of closed-loop metrics across bandwidth ----
    bandwidths = np.array([1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0])
    rows = []
    for w in bandwidths:
        rows.append((w,
                     metric_peak_y(w, "implemented"),
                     metric_peak_y(w, "exact"),
                     metric_rms_e3(w, "implemented"),
                     metric_rms_e3(w, "exact")))
    print("omega_o | peak|y| imp | peak|y| ex  | RMS e3 imp | RMS e3 ex")
    for w, pyi, pye, ei, ee in rows:
        print(f"{w:6.1f} | {pyi:10.4f} | {pye:9.4f} | {ei:10.4f} | {ee:9.4f}")

    # spectral radii sanity check
    print("\nspectral radius (audited vs exact):")
    for w in (1.0, 5.0, 20.0, 40.0, 100.0):
        gi = observer_gains(w, "implemented")
        ge = observer_gains(w, "exact")
        from analysis_core import error_matrix_from_gains
        rho_i = max(abs(np.linalg.eigvals(error_matrix_from_gains(gi, TS))))
        rho_e = max(abs(np.linalg.eigvals(error_matrix_from_gains(ge, TS))))
        print(f"  w={w:6.1f}: audited SR={rho_i:.4f}  exact SR={rho_e:.4f}  target rho={np.exp(-w*TS):.4f}")

    draw_figure(bandwidths, rows)


def draw_figure(bandwidths, rows):
    w_mod = 30.0
    w_high = 100.0
    t_m, _, e3_imp_m, _, _ = simulate(w_mod, "implemented", t_end=6.0)
    _, _, e3_ex_m, _, _ = simulate(w_mod, "exact", t_end=6.0)
    t_h, _, e3_imp_h, _, _ = simulate(w_high, "implemented", t_end=12.0)
    _, _, e3_ex_h, _, _ = simulate(w_high, "exact", t_end=12.0)

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.5),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.42})

    # (a) disturbance-estimation error at moderate bandwidth
    ax = axes[0]
    ax.plot(t_m, e3_imp_m, color=ORANGE, lw=1.5, label="implemented")
    ax.plot(t_m, e3_ex_m, color=NAVY, ls="--", lw=1.5, label="exact poles")
    ax.axvline(STEP_TIME, color=GREY, ls=":", lw=0.9)
    ax.set(xlabel="time (s)", ylabel=r"estimation error $e_3$ (m s$^{-2}$)",
           xlim=(0, 6.0))
    ax.legend(frameon=False, loc="upper right", handlelength=2.6)
    style_axis(ax, grid="y")
    panel_label(ax, "a")

    # (b) disturbance-estimation error at high bandwidth: audited rings, exact dead-beat
    ax = axes[1]
    ax.plot(t_h, e3_imp_h, color=ORANGE, lw=1.2)
    ax.plot(t_h, e3_ex_h, color=NAVY, ls="--", lw=1.5)
    ax.axvline(STEP_TIME, color=GREY, ls=":", lw=0.9)
    ax.set(xlabel="time (s)", ylabel=r"estimation error $e_3$ (m s$^{-2}$)",
           xlim=(0, 12.0))
    style_axis(ax, grid="y")
    panel_label(ax, "b")

    # (c) RMS post-step disturbance error vs bandwidth: the rebound
    ax = axes[2]
    w = np.array([r[0] for r in rows])
    ei = np.array([r[3] for r in rows])
    ee = np.array([r[4] for r in rows])
    ax.semilogx(w, ei, "o-", color=ORANGE, label="implemented")
    ax.semilogx(w, ee, "s--", color=NAVY, markerfacecolor="white", label="exact poles")
    jmin = int(np.argmin(ei))
    ax.plot(w[jmin], ei[jmin], "o", color=ORANGE, markeredgecolor="white",
            markeredgewidth=0.8, zorder=4)
    ax.annotate("rebound", xy=(w[jmin], ei[jmin]), xytext=(w[jmin] * 1.8, ei[jmin] * 1.6),
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8}, color=GREY, fontsize=7)
    ax.set(xlabel=r"observer bandwidth $\omega_o$ (rad s$^{-1}$)",
           ylabel=r"RMS disturbance error $e_3$ (m s$^{-2}$)")
    ax.legend(frameon=False, loc="upper right", handlelength=2.4)
    style_axis(ax, grid="both")
    panel_label(ax, "c")

    save_figure(fig, "fig_closed_loop")
    print("saved figures/fig_closed_loop.*")


if __name__ == "__main__":
    main()
