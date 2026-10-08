import numpy as np

import validate_theory as vt


def test_log_uniform_amplitudes_include_importance_weight():
    comp = vt.generate_wave_components(3, N=32, seed=42)
    dlog = np.log(2.4 / 0.6)
    expected_m0 = np.sum(
        comp["freq"] * vt.jonswap_s(comp["freq"], comp["w0"]) * dlog / 32
    )
    actual_m0 = np.sum(comp["amp_raw"] ** 2) / 2
    assert np.isclose(actual_m0, expected_m0, rtol=1e-12, atol=0)


def test_physical_band_share_uses_linear_frequency_measure():
    share = vt.physical_band_share(1.0, 0.6, 2.4)
    assert np.isclose(share, 0.9757493, rtol=3e-5)


def test_exact_zoh_gains_place_triple_pole():
    dt = 0.05
    for bandwidth in (2.0, 6.0, 20.0):
        gains = vt.exact_zoh_poleplacement_gains(bandwidth, dt)
        matrix = vt.error_matrix_from_gains(gains, dt)
        rho = np.exp(-bandwidth * dt)
        assert np.allclose(np.poly(matrix), np.poly([rho] * 3), atol=1e-10)


def test_implemented_gain_rule_matches_the_transcribed_constants():
    """Pins the transcribed rule of Eq. (3): (3a, 3a^2/Ts, 2a^3/Ts^2).

    This test is the checkable half of the provenance statement. The original
    implementation is not redistributed, so the transcription itself -- not the
    source it came from -- is what is held fixed here. The spurious factor two
    in the third gain is visible as the literal 2.0 below.
    """
    dt = 0.05
    for bandwidth in (0.5, 2.0, 6.0, 20.0, 100.0):
        rho = np.exp(-bandwidth * dt)
        a = 1.0 - rho
        expected = np.array([3.0 * a, 3.0 * a**2 / dt, 2.0 * a**3 / dt**2])
        assert np.array_equal(vt.implemented_gains(bandwidth, dt), expected)


def test_disturbance_jump_enters_extended_state_only():
    assert np.array_equal(vt.disturbance_jump_vector(), np.array([0.0, 0.0, 1.0]))


def test_jump_steady_state_matches_analytic_expression():
    dt = 0.05
    bandwidth = 6.0
    matrix = vt.implemented_error_matrix(bandwidth, dt)
    response = np.linalg.solve(
        np.eye(3) - matrix, vt.disturbance_jump_vector()
    )
    rho = np.exp(-bandwidth * dt)
    assert np.isclose(response[2], 3.0 / (2.0 * (1.0 - rho)), rtol=1e-12)


def test_white_noise_covariance_satisfies_discrete_lyapunov_equation():
    matrix = vt.implemented_error_matrix(6.0, 0.05)
    gains = vt.implemented_gains(6.0, 0.05)
    covariance = vt.stationary_noise_covariance(matrix, gains, sigma=0.01)
    residual = covariance - matrix @ covariance @ matrix.T - 0.01**2 * np.outer(gains, gains)
    assert np.max(np.abs(residual)) < 1e-11


def test_impulse_peak_is_smaller_than_bounded_noise_gain():
    metrics = vt.noise_metrics(6.0, 0.05, horizon=20000)
    assert metrics["e3_linf_induced"] > metrics["e3_impulse_peak"]
    assert metrics["e3_h2"] > 0


def test_frequency_domain_case_study_is_normalized_and_finite():
    omega_p = 2 * np.pi / 7.5
    omega, spectrum = vt.normalized_jonswap_grid(omega_p, nodes=2001)
    assert np.isclose(np.trapezoid(spectrum, omega), 1.0, atol=1e-10)
    gains = vt.disturbance_error_gain(2.0, np.array([0.5, 1.0]), 0.05)
    assert np.all(np.isfinite(gains))
    assert np.all(gains >= 0)


def test_tuning_criterion_combines_rms_variances():
    criterion = vt.tuning_criterion(
        omega_o=2.0,
        omega_p=2 * np.pi / 7.5,
        sigma_d=0.025,
        sigma_n=0.01,
        dt=0.05,
    )
    expected = np.hypot(criterion["disturbance_rms"], criterion["noise_rms"])
    assert np.isclose(criterion["total_rms"], expected)
    assert criterion["total_rms"] > 0


def test_exact_design_can_be_evaluated_with_the_same_metrics():
    implemented = vt.noise_metrics(6.0, 0.05, horizon=20000, design="implemented")
    exact = vt.noise_metrics(6.0, 0.05, horizon=20000, design="exact")
    assert implemented["e3_h2"] > 0
    assert exact["e3_h2"] > 0
    assert not np.isclose(implemented["e3_h2"], exact["e3_h2"])
