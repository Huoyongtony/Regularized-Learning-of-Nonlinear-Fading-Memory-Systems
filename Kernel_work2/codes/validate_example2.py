"""Independent numerical audit of thesis Example 2.

Run from any directory. Results go to output/example2_validation by default.
No thesis source, notebook, or existing figure is changed.
Requires NumPy, plus ReportLab to import the legacy reconstruction for the
final cross-check. For the approved sinusoidal-only example, run
reproduce_example2_sinusoids.py instead.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


T = 30.0
HISTORY = 4.0
LAMBDA = 0.5
GAMMA = 14.8
OMEGA = np.exp(np.linspace(-1.0, 3.0, 100))


def sine_response(t: np.ndarray) -> np.ndarray:
    """Analytic zero-state response of 10/((s+1)(s+2))."""
    omega = OMEGA[:, None]
    time = np.asarray(t)[None, :]
    s, c = np.sin(omega * time), np.cos(omega * time)
    return 10 * (
        (s - omega * c + omega * np.exp(-time)) / (1 + omega**2)
        - (2 * s - omega * c + omega * np.exp(-2 * time)) / (4 + omega**2)
    )


def grid(dt: float) -> np.ndarray:
    count = round(HISTORY / dt)
    if not math.isclose(count * dt, HISTORY, abs_tol=1e-12):
        raise ValueError("Time step must divide the history length.")
    return np.linspace(-HISTORY, 0.0, count + 1)


def fit(
    x: np.ndarray, target: np.ndarray, q: np.ndarray, rho: float,
    gamma: float = GAMMA,
) -> dict:
    """Modified bilinear kernel, a=b=1; both gain and squared gain reported."""
    gram = (x * q) @ x.T + np.outer(x[:, -1], x[:, -1])
    alpha = np.linalg.solve(gram + gamma * np.eye(len(target)), target)
    representer = alpha @ x
    beta2 = float(alpha @ gram @ alpha)
    primal_beta2 = float(np.sum(q * representer**2) + representer[-1]**2)
    c2_full = 1 / rho
    c2_window = float(np.sum(q))
    residual = (gram + gamma * np.eye(len(target))) @ alpha - target
    # This FFT scans all discrete frequencies for the implemented finite-delay
    # quadrature model. It is a lower estimate of the peak, not a proof of it.
    taps = (q * representer)[::-1].copy()
    taps[0] += representer[-1]
    nfft = 1 << max(18, (8 * len(taps) - 1).bit_length())
    frequency_response = np.fft.rfft(taps, n=nfft)
    peak_index = int(np.argmax(np.abs(frequency_response)))
    recovered = (1 - frequency_response) / (1 + frequency_response)
    return {
        "beta": math.sqrt(beta2),
        "beta_squared": beta2,
        "c_squared_full": c2_full,
        "c_squared_quadrature_window": c2_window,
        "bound_full": math.sqrt(beta2 * (1 + c2_full)),
        "bound_squared_full": beta2 * (1 + c2_full),
        "bound_window": math.sqrt(beta2 * (1 + c2_window)),
        "primal_dual_norm_error": abs(beta2 - primal_beta2),
        "relative_solve_residual": float(np.linalg.norm(residual) / np.linalg.norm(target)),
        "gram_min_eigenvalue": float(np.linalg.eigvalsh(gram)[0]),
        "sampled_discrete_peak_gain": float(np.abs(frequency_response[peak_index])),
        "sampled_minimum_recovered_real_part": float(np.min(recovered.real)),
        "fft_points": nfft,
        "training_points": len(target),
    }


def discrete_experiment(
    dt: float, multiplier: float, impulse: str = "unit_sample",
    quadrature: str = "trapezoid", gamma: float = GAMMA,
    ode_y: np.ndarray | None = None,
) -> dict:
    tau = grid(dt)
    u = np.sin(OMEGA[:, None] * (T + tau)[None, :])
    y = sine_response(T + tau) if ode_y is None else ode_y
    x, target = 0.5 * (u + y), 0.5 * (u[:, -1] - y[:, -1])
    if impulse != "none":
        pulse = np.zeros(len(tau))
        pulse[-1] = 1.0 if impulse == "unit_sample" else 1.0 / dt
        x = np.vstack((x, 0.5 * pulse))
        target = np.append(target, 0.5 * pulse[-1])
    weights = np.full(len(tau), dt)
    if quadrature == "trapezoid":
        weights[[0, -1]] *= 0.5
    elif quadrature != "rectangle":
        raise ValueError(quadrature)
    rho = multiplier * LAMBDA
    q = weights * np.exp(rho * tau)
    result = fit(x, target, q, rho, gamma)
    result.update(dt=dt, exponent_multiplier=multiplier, impulse=impulse,
                  quadrature=quadrature, gamma=gamma)
    return result


def exact_continuous_experiment(multiplier: float) -> dict:
    """Integrate every Gram entry analytically; no time discretization.

    A unit endpoint sample becomes an independent point-evaluation datum:
    zero history a.e. and value 1 at tau=0. This is the dt->0 limit of the
    specified unit-height sample, not a continuous-time Dirac impulse.
    """
    omega = OMEGA
    sin_coeff = 0.5 * (1 + 10 * (1 / (1 + omega**2) - 2 / (4 + omega**2)))
    cos_coeff = 5 * (-omega / (1 + omega**2) + omega / (4 + omega**2))
    positive = 0.5 * (cos_coeff - 1j * sin_coeff) * np.exp(1j * omega * T)
    coeff = np.column_stack((positive, positive.conj(),
                             5 * omega / (1 + omega**2) * np.exp(-T),
                             -5 * omega / (4 + omega**2) * np.exp(-2 * T)))
    rates = np.column_stack((1j * omega, -1j * omega,
                             np.full(100, -1), np.full(100, -2)))
    rho = multiplier * LAMBDA
    integral = np.zeros((100, 100), dtype=complex)
    for j in range(4):
        for k in range(4):
            p = rates[:, j, None] + rates[None, :, k] + rho
            factor = np.empty_like(p)
            nonzero = np.abs(p) > 1e-13
            np.divide(-np.expm1(-p * HISTORY), p, out=factor, where=nonzero)
            factor[~nonzero] = HISTORY
            integral += coeff[:, j, None] * coeff[None, :, k] * factor
    x0 = coeff.sum(axis=1).real
    endpoint = np.append(x0, 0.5)
    gram = np.zeros((101, 101))
    gram[:100, :100] = integral.real
    gram += np.outer(endpoint, endpoint)
    u0 = np.sin(omega * T)
    target = np.append(u0 - x0, 0.5)
    alpha = np.linalg.solve(gram + GAMMA * np.eye(101), target)
    beta2 = float(alpha @ gram @ alpha)
    # Verify the exponential representation against the analytic real formula.
    probe = np.array([-4.0, -1.3, 0.0])
    represented = np.sum(coeff[:, :, None] * np.exp(rates[:, :, None] * probe), axis=1)
    expected = 0.5 * (np.sin(omega[:, None] * (T + probe)) + sine_response(T + probe))
    error = float(np.max(np.abs(represented - expected)))
    assert error < 1e-11, error
    return {
        "exponent_multiplier": multiplier,
        "beta": math.sqrt(beta2),
        "c_squared_full": 1 / rho,
        "c_squared_window_exact": -math.expm1(-rho * HISTORY) / rho,
        "bound_full": math.sqrt(beta2 * (1 + 1 / rho)),
        "bound_window": math.sqrt(beta2 * (1 - math.expm1(-rho * HISTORY) / rho)),
        "max_gram_imaginary_part": float(np.max(np.abs(integral.imag))),
        "representation_check_error": error,
    }


def rk4_responses(dt: float) -> np.ndarray:
    """Independent ODE simulation: v'=-v+u, z'=v-2z, y=10z."""
    count = round(T / dt)
    history_count = round(HISTORY / dt)
    state = np.zeros((2, len(OMEGA)))
    outputs = np.empty((len(OMEGA), history_count + 1))

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        return np.array((-x[0] + np.sin(OMEGA * t), x[0] - 2 * x[1]))

    for step in range(count + 1):
        t = step * dt
        if step >= count - history_count:
            outputs[:, step - count + history_count] = 10 * state[1]
        if step == count:
            break
        k1 = rhs(t, state)
        k2 = rhs(t + dt / 2, state + dt / 2 * k1)
        k3 = rhs(t + dt / 2, state + dt / 2 * k2)
        k4 = rhs(t + dt, state + dt * k3)
        state += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "output/example2_validation")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    convergence = [discrete_experiment(dt, multiplier)
                   for multiplier in (1.0, 2.0)
                   for dt in (0.008, 0.004, 0.002, 0.001, 0.0005, 0.00025)]
    sensitivity = [discrete_experiment(0.002, multiplier, impulse, quadrature)
                   for multiplier in (1.0, 2.0)
                   for impulse in ("unit_sample", "none", "unit_area_sample")
                   for quadrature in ("trapezoid", "rectangle")]
    exact = [exact_continuous_experiment(multiplier) for multiplier in (1.0, 2.0)]
    numerical_y = rk4_responses(0.002)
    ode_error = float(np.max(np.abs(numerical_y - sine_response(T + grid(0.002)))))
    ode_result = discrete_experiment(0.002, 2.0, ode_y=numerical_y)
    baseline = next(r for r in convergence if r["dt"] == 0.002 and r["exponent_multiplier"] == 2)

    # Compare to the pre-existing reconstruction only after independent fits.
    old_path = Path(__file__).with_name("reproduce_linear_passivity.py")
    spec = importlib.util.spec_from_file_location("existing_reconstruction", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    _, _, _, _, old_diag = old.learn_scattered_functional(0.002, True, 2.0, GAMMA)
    old_difference = abs(baseline["bound_full"] - old_diag["certified_gain_full_history"])

    assert ode_error < 1e-7, ode_error
    assert abs(ode_result["bound_full"] - baseline["bound_full"]) < 1e-8
    assert old_difference < 1e-10, old_difference
    assert all(r["primal_dual_norm_error"] < 1e-12 for r in convergence)
    assert all(r["relative_solve_residual"] < 1e-12 for r in convergence)
    assert all(r["bound_full"] < 1 for r in convergence if r["exponent_multiplier"] == 2)
    assert all(r["bound_full"] > 1 for r in convergence if r["exponent_multiplier"] == 1)
    # This also rules out agreement caused solely by one quadrature grid.
    for reference in exact:
        refined = next(r for r in convergence if r["dt"] == 0.00025
                       and r["exponent_multiplier"] == reference["exponent_multiplier"])
        assert abs(refined["bound_full"] - reference["bound_full"]) < 1e-5

    results = {
        "parameters": {"plant": "10/((s+1)(s+2))", "duration": T,
                       "history": HISTORY, "lambda": LAMBDA, "gamma": GAMMA,
                       "a": 1, "b": 1, "sine_count": len(OMEGA)},
        "baseline": baseline,
        "exact_continuous": exact,
        "convergence": convergence,
        "sensitivity": sensitivity,
        "checks": {"rk4_max_output_error": ode_error,
                   "rk4_gain_bound": ode_result["bound_full"],
                   "difference_from_existing_reconstruction": old_difference,
                   "all_assertions_passed": True},
        "interpretation": [
            "The 0.999 value is an upper bound on gain, not the squared gain or measured gain.",
            "Multiplier 2 means w(tau)=exp(lambda*tau), with density exp(2*lambda*tau).",
            "Multiplier 1 means w(tau)=exp(lambda*tau/2), as printed in the thesis.",
            "The reproduced dataset uses a unit-height terminal sample, not a unit-area Dirac impulse.",
            "The certificate uses full-history c^2; the learned histories have zero prehistory before -4s.",
            "Finite-grid FFT peaks are diagnostic samples, not certified global peak gains.",
            "Agreement reconstructs the stated numerical experiment; it does not recover the missing original run.",
        ],
    }
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("convergence", convergence), ("sensitivity", sensitivity)):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["Example 2: independent numerical validation", "",
             "Plant: G(s)=10/((s+1)(s+2)); zero initial state.",
             "Training: 100 unit-amplitude zero-phase sines; omega=exp(linspace(-1,3,100)) rad/s.",
             "Duration: 30 s; history: [26,30] s; prehistory before 26 s is zeroed in the kernel.",
             "Additional datum: terminal sample of height 1 at t=30 s, with simultaneous plant output 0.",
             "Scattering: x=(u+y)/2, target=(u(30)-y(30))/2.",
             "Kernel: integral x(tau)z(tau)w(tau)^2 d tau + x(0)z(0), with a=b=1.",
             "Regularization: gamma=14.8; lambda=0.5; baseline dt=0.002 s, trapezoid quadrature.",
             "Certificate: beta=sqrt(alpha^T K alpha); gain <= beta*sqrt(b+a*c^2).", "",
             f"Baseline gain upper bound: {baseline['bound_full']:.12f}",
             f"Baseline squared gain upper bound: {baseline['bound_squared_full']:.12f}",
             f"Baseline RKHS norm beta: {baseline['beta']:.12f}",
             f"Baseline sampled discrete peak gain: {baseline['sampled_discrete_peak_gain']:.12f}",
             f"Independent RK4 maximum output discrepancy: {ode_error:.3e}",
             f"Independent RK4 gain bound: {ode_result['bound_full']:.12f}", "",
             "Analytic continuous-history integration (unit endpoint point datum):"]
    for row in exact:
        lines.append(f"  exponent multiplier {row['exponent_multiplier']:g}: "
                     f"bound={row['bound_full']:.12f}, c^2={row['c_squared_full']:g}")
    lines += ["", "Time-step convergence (trapezoid, unit-height terminal sample):",
              "multiplier  dt          gain bound"]
    for row in convergence:
        lines.append(f"{row['exponent_multiplier']:10g}  {row['dt']:<10g}  {row['bound_full']:.12f}")
    lines += ["", "Sensitivity at dt=0.002:", "multiplier  impulse            quadrature  gain bound"]
    for row in sensitivity:
        lines.append(f"{row['exponent_multiplier']:10g}  {row['impulse']:<18} "
                     f"{row['quadrature']:<11} {row['bound_full']:.12f}")
    lines += ["", "Interpretation and limits:"] + ["- " + x for x in results["interpretation"]]
    lines += ["", "All numerical checks passed.", "",
              "Conclusion: 0.999 is a valid rounded upper bound for the reproduced specification",
              "with w(tau)=exp(0.5*tau), c^2=1, and a unit-height terminal sample.",
              "The printed w(tau)=exp(0.25*tau) convention instead gives a bound above 1.",
              "A bound above 1 does not itself prove failure of passivity; it fails this sufficient test.",
              "Do not call the terminal sample a continuous-time unit-area impulse.",
              "This validation does not establish provenance of the original thesis figure."]
    report = "\n".join(lines) + "\n"
    (output / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
