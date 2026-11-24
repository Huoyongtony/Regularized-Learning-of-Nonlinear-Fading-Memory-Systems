"""Reproduce thesis Example 2 using 100 sinusoidal training inputs.

Run with NumPy and Matplotlib installed. The independent numerical checks
also use the NumPy-only functions in validate_example2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import validate_example2 as independent


ROOT = Path(__file__).resolve().parents[2]
DURATION = 30.0
HISTORY = 4.0
LAMBDA = 1.0
GAMMA = 14.8
OMEGA = np.exp(np.linspace(-1.0, 3.0, 100))


def plant_response(time):
    """Zero-state response of G(s)=10/((s+1)(s+2))."""
    omega = OMEGA[:, None]
    time = np.asarray(time)[None, :]
    return 10 * sum(
        sign * (rate*np.sin(omega*time) - omega*np.cos(omega*time)
                + omega*np.exp(-rate*time)) / (rate**2 + omega**2)
        for rate, sign in ((1, 1), (2, -1))
    )


def fit_grid(dt, outputs=None):
    tau = np.linspace(-HISTORY, 0, round(HISTORY/dt)+1)
    time = DURATION + tau
    u = np.sin(OMEGA[:, None]*time[None, :])
    y = plant_response(time) if outputs is None else outputs
    x = (u+y)/2
    target = (u[:, -1]-y[:, -1])/2
    q = np.full(tau.size, dt)
    q[[0, -1]] /= 2
    q *= np.exp(LAMBDA*tau)
    gram = (x*q) @ x.T + np.outer(x[:, -1], x[:, -1])
    coefficients = np.linalg.solve(gram+GAMMA*np.eye(100), target)
    representer = coefficients @ x
    beta2 = float(coefficients @ gram @ coefficients)
    primal = float(np.sum(q*representer**2)+representer[-1]**2)
    residual = (gram+GAMMA*np.eye(100)) @ coefficients-target
    diagnostics = {
        "dt": dt,
        "training_pairs": len(target),
        "history_samples": tau.size,
        "beta_squared": beta2,
        "beta": float(np.sqrt(beta2)),
        "c_squared": 1/LAMBDA,
        "gain_bound": float(np.sqrt(beta2*(1+1/LAMBDA))),
        "squared_gain_bound": beta2*(1+1/LAMBDA),
        "quadrature_history_mass": float(q.sum()),
        "primal_dual_norm_error": abs(beta2-primal),
        "relative_solve_residual": float(np.linalg.norm(residual)/np.linalg.norm(target)),
    }
    return tau, q, representer, coefficients, diagnostics


def exact_history_fit():
    """Analytic integration of all Gram entries, with no time grid."""
    omega = OMEGA
    sine = .5*(1+10/(1+omega**2)-20/(4+omega**2))
    cosine = 5*(-omega/(1+omega**2)+omega/(4+omega**2))
    positive = .5*(cosine-1j*sine)*np.exp(1j*omega*DURATION)
    coefficients = np.column_stack((
        positive, positive.conj(),
        5*omega/(1+omega**2)*np.exp(-DURATION),
        -5*omega/(4+omega**2)*np.exp(-2*DURATION),
    ))
    rates = np.column_stack((1j*omega, -1j*omega,
                             np.full(100, -1), np.full(100, -2)))
    integral = np.zeros((100, 100), dtype=complex)
    for j in range(4):
        for k in range(4):
            rate = rates[:, j, None]+rates[None, :, k]+LAMBDA
            factor = np.full_like(rate, HISTORY)
            nonzero = np.abs(rate)>1e-13
            np.divide(-np.expm1(-rate*HISTORY), rate, out=factor, where=nonzero)
            integral += coefficients[:, j, None]*coefficients[None, :, k]*factor
    endpoint = coefficients.sum(axis=1).real
    target = np.sin(omega*DURATION)-endpoint
    gram = integral.real+np.outer(endpoint, endpoint)
    alpha = np.linalg.solve(gram+GAMMA*np.eye(100), target)
    beta2 = float(alpha @ gram @ alpha)
    probe = np.array([-4., -1.3, 0.])
    represented = np.sum(coefficients[:, :, None]
                         *np.exp(rates[:, :, None]*probe), axis=1)
    expected = .5*(np.sin(omega[:, None]*(DURATION+probe))
                   +plant_response(DURATION+probe))
    error = float(np.max(np.abs(represented-expected)))
    assert error<1e-11, error
    return {"beta_squared": beta2,
            "gain_bound": float(np.sqrt(beta2*(1+1/LAMBDA))),
            "representation_error": error,
            "maximum_imaginary_gram_entry": float(np.max(np.abs(integral.imag)))}


def frequency_responses(omega, tau, q, representer):
    scattered = np.exp(1j*np.outer(omega, tau)) @ (q*representer)
    scattered += representer[-1]
    learned = (1-scattered)/(1+scattered)
    original = 10/((1j*omega+1)*(1j*omega+2))
    return original, scattered, learned


def save_figure(output, original, learned):
    # Match the existing figure canvas so the thesis layout is preserved.
    fig, ax = plt.subplots(figsize=(336/72, 240/72))
    for values, color, label in ((learned, "#149ee7", "Passive approximation"),
                                  (original, "#e86f43", "Original system")):
        ax.plot(values.real, values.imag, "o", ms=3.5, mfc=color,
                mec="black", mew=.25, label=label)
    ax.set(xlim=(-.7, 4.25), ylim=(-3.25, .1), xlabel="Real", ylabel="Imaginary",
           title="Nyquist plot", xticks=[0, 1, 2, 3, 4], yticks=[0, -1, -2, -3])
    ax.set_xlabel("Real", fontsize=11)
    ax.set_ylabel("Imaginary", fontsize=11)
    ax.set_title("Nyquist plot", fontsize=15)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", fontsize=10, framealpha=1, edgecolor="black")
    fig.subplots_adjust(left=.15, right=.98, bottom=.16, top=.87)
    fig.savefig(output/"scatter_linear_dots.pdf")
    fig.savefig(output/"scatter_linear_dots.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT/"output/example2_sinusoids_only")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    convergence = []
    for dt in (.008, .004, .002, .001, .0005, .00025):
        _, _, _, _, row = fit_grid(dt)
        convergence.append(row)
        assert row["training_pairs"]==100
        assert row["gain_bound"]<.994
        assert row["relative_solve_residual"]<1e-12
        assert row["primal_dual_norm_error"]<1e-12
    tau, q, representer, coefficients, baseline = fit_grid(.002)
    exact = exact_history_fit()
    numerical_outputs = independent.rk4_responses(.002)
    ode_error = float(np.max(np.abs(numerical_outputs-plant_response(DURATION+tau))))
    _, _, _, _, ode = fit_grid(.002, numerical_outputs)
    separate_fit = independent.discrete_experiment(.002, 2., impulse="none")
    assert ode_error<1e-7
    assert abs(ode["gain_bound"]-baseline["gain_bound"])<1e-8
    assert abs(separate_fit["bound_full"]-baseline["gain_bound"])<1e-10
    assert abs(convergence[-1]["gain_bound"]-exact["gain_bound"])<1e-5
    original, scattered, learned = frequency_responses(OMEGA, tau, q, representer)
    assert np.max(np.abs(scattered))<baseline["gain_bound"]
    assert np.min(learned.real)>0
    results = {
        "parameters": {"plant": "10/((s+1)(s+2))", "duration": DURATION,
                       "history_duration": HISTORY, "lambda": LAMBDA,
                       "weight": "exp(lambda*tau/2)", "gamma": GAMMA,
                       "a": 1, "b": 1, "initial_state": "zero",
                       "training_inputs": "100 unit-amplitude zero-phase sinusoids",
                       "training_angular_frequencies": OMEGA.tolist(),
                       "training_pairs": 100, "dt": .002,
                       "quadrature": "trapezoid"},
        "baseline": baseline, "exact_continuous_history": exact,
        "convergence": convergence,
        "checks": {"rk4_maximum_output_error": ode_error,
                   "rk4_gain_bound": ode["gain_bound"],
                   "independent_fit_difference": abs(separate_fit["bound_full"]-baseline["gain_bound"]),
                   "sampled_scattered_peak_gain": float(np.max(np.abs(scattered))),
                   "sampled_minimum_recovered_real_part": float(np.min(learned.real)),
                   "all_checks_passed": True},
    }
    np.savez_compressed(output/"model_and_responses.npz", frequencies=OMEGA,
                        tau=tau, integral_weights=q, representer=representer,
                        coefficients=coefficients, original=original,
                        scattered=scattered, learned=learned)
    (output/"results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    lines = ["Example 2 with sinusoidal training inputs only", "",
             "Plant: 10/((s+1)(s+2)), with zero initial state.",
             "Inputs: 100 unit-amplitude zero-phase sinusoids, omega=exp(linspace(-1,3,100)) rad/s.",
             "Each pair uses the transformed history over [26,30] s and its target at 30 s.",
             "Weight: w(tau)=exp(tau/2), lambda=1, c^2=1. Kernel a=b=1. Gamma=14.8.",
             "Baseline: dt=0.002 s, 2001 history samples, trapezoidal quadrature.", "",
             f"RKHS norm beta: {baseline['beta']:.12f}",
             f"Full-history gain upper bound: {baseline['gain_bound']:.12f}",
             f"Exact continuous-history gain upper bound: {exact['gain_bound']:.12f}",
             f"Independent RK4 maximum output discrepancy: {ode_error:.3e}",
             f"Independent RK4 gain upper bound: {ode['gain_bound']:.12f}", "",
             "Grid refinement: dt, full-history gain upper bound"]
    lines += [f"{row['dt']:.6f}, {row['gain_bound']:.12f}" for row in convergence]
    lines += ["", "All checks passed. The conservative gain bound 0.994 is valid.",
              "The gain bound is below 1, so scattering recovers an incrementally passive model.",
              "The Nyquist points are evaluated from the same fitted model used for the bound."]
    report = "\n".join(lines)+"\n"
    (output/"report.txt").write_text(report, encoding="utf-8")
    save_figure(output, original, learned)
    print(report)


if __name__=="__main__":
    main()
