"""Verified numerical core for the ZOH-discretized ESO paper.

The functions in this module use a physically defined angular-frequency
spectrum, an explicit sample-boundary disturbance jump, and performance
metrics whose input/output norms are stated separately.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import solve_discrete_lyapunov

G = 9.81
GAMMA_J = 3.3
ALPHA_J = 0.0081
SPREAD_DEG = 30.0
WAVE_COUNT = 32
SEED = 42
DT = 0.05

HS_TABLE = np.array([0.0, 0.1, 0.5, 1.25, 2.5, 4.0])
TP_TABLE = np.array([1.0, 3.0, 5.0, 7.5, 9.5, 12.0])


def jonswap_s(omega: np.ndarray | float, omega_p: float) -> np.ndarray:
    """One-sided JONSWAP density with respect to angular frequency."""
    omega_array = np.asarray(omega, dtype=float)
    ratio = omega_array / omega_p
    sigma = np.where(ratio < 1.0, 0.07, 0.09)
    peak = np.exp(-((ratio - 1.0) ** 2) / (2.0 * sigma**2))
    return (
        ALPHA_J
        * G**2
        / omega_array**5
        * np.exp(-1.25 * ratio**-4)
        * GAMMA_J**peak
    )


def physical_spectral_moment(
    omega_p: float,
    lower_ratio: float = 0.01,
    upper_ratio: float = 12.0,
    nodes: int = 200_001,
) -> float:
    omega = np.linspace(lower_ratio * omega_p, upper_ratio * omega_p, nodes)
    return float(trapezoid(jonswap_s(omega, omega_p), omega))


def physical_band_share(
    omega_p: float,
    lower_ratio: float = 0.6,
    upper_ratio: float = 2.4,
) -> float:
    band = physical_spectral_moment(omega_p, lower_ratio, upper_ratio)
    full = physical_spectral_moment(omega_p)
    return band / full


def generate_wave_components(
    sea_state: int, N: int = WAVE_COUNT, seed: int = SEED
) -> dict[str, np.ndarray | float]:
    """Draw log-uniform components with correct importance weights.

    If q(omega)=1/(DeltaLog*omega), the unbiased Monte Carlo estimate of
    integral S(omega)domega is DeltaLog/N * sum omega_i*S(omega_i).
    """
    hs = float(HS_TABLE[sea_state])
    tp = float(TP_TABLE[sea_state])
    omega_p = 2.0 * np.pi / max(tp, 0.5)
    rng = np.random.default_rng(seed)
    log_lo = np.log(0.6 * omega_p)
    log_hi = np.log(2.4 * omega_p)
    delta_log = log_hi - log_lo
    frequency = np.exp(rng.uniform(log_lo, log_hi, N))
    spectrum = jonswap_s(frequency, omega_p)
    amplitude_raw = np.sqrt(2.0 * frequency * spectrum * delta_log / N)

    raw_m0 = float(np.sum(amplitude_raw**2) / 2.0)
    scale = hs / (4.0 * np.sqrt(raw_m0)) if raw_m0 > 0 else 1.0
    amplitude = amplitude_raw * scale
    direction_offset = rng.uniform(
        -0.5 * np.deg2rad(SPREAD_DEG),
        0.5 * np.deg2rad(SPREAD_DEG),
        N,
    )
    direction = np.deg2rad(30.0) + direction_offset
    phase = rng.uniform(0.0, 2.0 * np.pi, N)
    return {
        "freq": frequency,
        "amp": amplitude,
        "amp_raw": amplitude_raw,
        "k": frequency**2 / G,
        "dir_angle": direction,
        "phase": phase,
        "hs": hs,
        "tp": tp,
        "w0": omega_p,
        "omega_p": omega_p,
        "scale": scale,
    }


def implemented_gains(omega_o: float, dt: float = DT) -> np.ndarray:
    rho = np.exp(-omega_o * dt)
    a = 1.0 - rho
    return np.array([3.0 * a, 3.0 * a**2 / dt, 2.0 * a**3 / dt**2])


def exact_zoh_poleplacement_gains(
    omega_o: float, dt: float = DT
) -> np.ndarray:
    rho = np.exp(-omega_o * dt)
    a = 1.0 - rho
    return np.array([3.0 * a, a**2 * (3.0 - 0.5 * a) / dt, a**3 / dt**2])


def error_matrix_from_gains(gains: np.ndarray, dt: float = DT) -> np.ndarray:
    l1, l2, l3 = np.asarray(gains, dtype=float)
    return np.array(
        [
            [1.0 - l1, dt, 0.5 * dt**2],
            [-l2, 1.0, dt],
            [-l3, 0.0, 1.0],
        ]
    )


def implemented_error_matrix(omega_o: float, dt: float = DT) -> np.ndarray:
    return error_matrix_from_gains(implemented_gains(omega_o, dt), dt)


def exact_error_matrix(omega_o: float, dt: float = DT) -> np.ndarray:
    return error_matrix_from_gains(exact_zoh_poleplacement_gains(omega_o, dt), dt)


def gains_for_design(
    omega_o: float, dt: float = DT, design: str = "implemented"
) -> np.ndarray:
    if design == "implemented":
        return implemented_gains(omega_o, dt)
    if design == "exact":
        return exact_zoh_poleplacement_gains(omega_o, dt)
    raise ValueError(f"unknown observer design: {design}")


def disturbance_jump_vector() -> np.ndarray:
    """Input vector for delta_k=d_{k+1}-d_k at a sample boundary."""
    return np.array([0.0, 0.0, 1.0])


def stationary_noise_covariance(
    matrix: np.ndarray, gains: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    """State covariance under independent, zero-mean measurement noise."""
    covariance_input = sigma**2 * np.outer(gains, gains)
    return solve_discrete_lyapunov(matrix, covariance_input)


def noise_metrics(
    omega_o: float,
    dt: float = DT,
    horizon: int = 20_000,
    design: str = "implemented",
) -> dict[str, float]:
    """Separate unit-noise H2, impulse-peak, and Linfinity metrics for e3."""
    gains = gains_for_design(omega_o, dt, design)
    matrix = error_matrix_from_gains(gains, dt)
    covariance = stationary_noise_covariance(matrix, gains, sigma=1.0)
    response = -gains.copy()
    e3_impulse = []
    for _ in range(horizon):
        e3_impulse.append(float(response[2]))
        response = matrix @ response
        if np.linalg.norm(response, ord=np.inf) < 1e-14:
            break
    impulse = np.asarray(e3_impulse)
    return {
        "e3_h2": float(np.sqrt(max(covariance[2, 2], 0.0))),
        "e3_impulse_peak": float(np.max(np.abs(impulse))),
        "e3_linf_induced": float(np.sum(np.abs(impulse))),
    }


def disturbance_error_gain(
    omega_o: float,
    omega: np.ndarray | float,
    dt: float = DT,
    design: str = "implemented",
) -> np.ndarray:
    """Magnitude from sampled disturbance d_k to e3_k.

    A sinusoidal d_k has jump delta_k=(z-1)d_k. The jump enters the third
    extended state, yielding (z-1) C(zI-A)^(-1)B_delta.
    """
    matrix = error_matrix_from_gains(gains_for_design(omega_o, dt, design), dt)
    output = np.array([[0.0, 0.0, 1.0]])
    jump = disturbance_jump_vector()
    omega_array = np.atleast_1d(np.asarray(omega, dtype=float))
    gain = np.empty_like(omega_array)
    identity = np.eye(3)
    for index, physical_frequency in enumerate(omega_array):
        z = np.exp(1j * physical_frequency * dt)
        transfer = (z - 1.0) * (output @ np.linalg.solve(z * identity - matrix, jump))[0]
        gain[index] = abs(transfer)
    return gain if np.ndim(omega) else gain[0]


def normalized_jonswap_grid(
    omega_p: float, lower_ratio: float = 0.01, upper_ratio: float = 12.0,
    nodes: int = 20_001,
) -> tuple[np.ndarray, np.ndarray]:
    omega = np.linspace(lower_ratio * omega_p, upper_ratio * omega_p, nodes)
    spectrum = jonswap_s(omega, omega_p)
    spectrum /= trapezoid(spectrum, omega)
    return omega, spectrum


def spectral_disturbance_error_factor(
    omega_o: float,
    omega_p: float,
    dt: float = DT,
    design: str = "implemented",
) -> float:
    """RMS e3 per RMS disturbance for a normalized JONSWAP proxy."""
    omega, spectrum = normalized_jonswap_grid(omega_p)
    gain = disturbance_error_gain(omega_o, omega, dt, design)
    return float(np.sqrt(trapezoid(gain**2 * spectrum, omega)))


def tuning_criterion(
    omega_o: float,
    omega_p: float,
    sigma_d: float,
    sigma_n: float,
    dt: float = DT,
    design: str = "implemented",
) -> dict[str, float]:
    disturbance_factor = spectral_disturbance_error_factor(
        omega_o, omega_p, dt, design
    )
    noise_factor = noise_metrics(
        omega_o, dt, horizon=8_000, design=design
    )["e3_h2"]
    disturbance_variance = (sigma_d * disturbance_factor) ** 2
    noise_variance = (sigma_n * noise_factor) ** 2
    return {
        "disturbance_rms": float(np.sqrt(disturbance_variance)),
        "noise_rms": float(np.sqrt(noise_variance)),
        "total_rms": float(np.sqrt(disturbance_variance + noise_variance)),
    }
