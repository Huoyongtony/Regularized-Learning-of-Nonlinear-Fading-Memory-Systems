"""Audit Example 4 without changing the thesis or its existing figure.

Requires NumPy, SciPy, Numba, and Matplotlib.
Kernel_work2/reference/example4_curves.npz contains vector paths extracted from
the original figure PDF. This audits a specified reconstruction, not the
missing historical notebook execution state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.linalg import cho_factor, cho_solve


ROOT = Path(__file__).resolve().parents[2]
SCALE = 0.0005
GAMMA = 0.0005625
A = B = 0.1
SIGMA = 1.0
N = 1000
T = 2.0
DT = T / N  # Notebook quadrature convention; simulation grid is inclusive.
TIME = np.linspace(0, T, N)
AMPLITUDES = 15 * np.arange(1, 11)
ENDPOINTS = np.arange(10, N + 1, 10) - 1


def plant(solver="analytic"):
    u = AMPLITUDES[:, None] * np.sin(10 * TIME)
    if solver == "analytic":
        x = AMPLITUDES[:, None] * (
            5*np.sin(10*TIME) - 10*np.cos(10*TIME) + 10*np.exp(-5*TIME)
        ) / 125
    elif solver == "euler":
        x = np.zeros_like(u)
        step = TIME[1] - TIME[0]
        for n in range(N-1):
            x[:, n+1] = x[:, n] + step*(-5*x[:, n] + u[:, n])
    elif solver == "rk4":
        x = np.zeros_like(u)
        step = TIME[1] - TIME[0]
        for n in range(N-1):
            t, state = TIME[n], x[:, n]
            k1 = -5*state + AMPLITUDES*np.sin(10*t)
            k2 = -5*(state+step*k1/2) + AMPLITUDES*np.sin(10*(t+step/2))
            k3 = -5*(state+step*k2/2) + AMPLITUDES*np.sin(10*(t+step/2))
            k4 = -5*(state+step*k3) + AMPLITUDES*np.sin(10*(t+step))
            x[:, n+1] = state + step*(k1+2*k2+2*k3+k4)/6
    else:
        raise ValueError(solver)
    return u, np.clip(x, -5, 5), x


def training_data(solver):
    u, y, x = plant(solver)
    xp, yp = (u+y)/2, (u-y)/2
    histories = np.array([np.pad(signal[:end+1], (N-end-1, 0))
                          for signal in xp for end in ENDPOINTS])
    target = yp[:, ENDPOINTS].ravel()
    return histories, target, u, y, x


def fit_case(name, solver, density_rate, normalization=False, weighted_point=False):
    histories, target, u, y, x = training_data(solver)
    weight = np.exp(density_rate/2 * (TIME-T))
    normalizer = math.sqrt(float(np.sum(weight**2)*DT)) if normalization else 1.0
    weight = weight / normalizer
    point_weight = weight[-1] if weighted_point else 1.0
    features = np.column_stack((
        histories*weight*np.sqrt(A*DT),
        histories[:, -1]*np.sqrt(B)*point_weight)) * SCALE
    norms = np.sum(features**2, axis=1)
    distance = np.maximum(norms[:, None]+norms[None, :]-2*features@features.T, 0)
    gram = np.exp(-distance/(2*SIGMA**2))
    scaled_target = SCALE*target
    coefficients = cho_solve(cho_factor(gram+GAMMA*np.eye(len(target)), lower=True),
                             scaled_target)
    beta2 = float(coefficients@gram@coefficients)
    full_c2 = 1/density_rate/normalizer**2
    discrete_c2 = float(np.sum(weight**2)*DT)
    point_factor = B*point_weight**2
    # Scaling BOTH input and output by SCALE cancels from the physical gain.
    squared_full_bound = beta2*(point_factor + A*full_c2)/SIGMA**2
    squared_discrete_bound = beta2*(point_factor+A*discrete_c2)/SIGMA**2
    residual = (gram+GAMMA*np.eye(len(target)))@coefficients-scaled_target
    result = {
        "name": name, "solver": solver, "density_rate": density_rate,
        "normalized_history_weight": normalization,
        "weighted_endpoint": weighted_point,
        "beta_squared": beta2,
        "c_squared_infinite_history": full_c2,
        "c_squared_quadrature_history": discrete_c2,
        "weight_at_zero": float(weight[-1]),
        "squared_bound_continuous_mass": squared_full_bound,
        "gain_bound_continuous_mass": math.sqrt(squared_full_bound),
        "squared_bound_discrete_mass": squared_discrete_bound,
        "gain_bound_discrete_mass": math.sqrt(squared_discrete_bound),
        "solve_relative_residual": float(np.linalg.norm(residual)/np.linalg.norm(scaled_target)),
        "physical_training_mean_absolute_error": float(np.mean(np.abs(gram@coefficients/SCALE-target))),
    }
    assert result["solve_relative_residual"] < 1e-9
    return result, histories, coefficients, weight, point_weight


def refined_continuous_history(points):
    """Check convergence without using the finite input buffer's rectangular sum."""
    endpoints = TIME[ENDPOINTS]
    amplitudes = np.repeat(AMPLITUDES, len(endpoints))[:, None]
    terminal_times = np.tile(endpoints, len(AMPLITUDES))[:, None]
    tau = np.linspace(-T, 0, points)
    quadrature = np.full(points, T/(points-1))
    quadrature[[0, -1]] *= 0.5
    times = np.maximum(terminal_times+tau, 0)
    u = amplitudes*np.sin(10*times)
    state = amplitudes*(5*np.sin(10*times)-10*np.cos(10*times)+10*np.exp(-5*times))/125
    y = np.clip(state, -5, 5)
    history = (u+y)/2
    target = SCALE*(u[:, -1]-y[:, -1])/2
    q = quadrature*np.exp(20*tau)
    features = np.column_stack((history*np.sqrt(A*q), history[:, -1]*np.sqrt(B)))*SCALE
    norms = np.sum(features**2, axis=1)
    gram = np.exp(-np.maximum(norms[:, None]+norms[None, :]-2*features@features.T, 0)/2)
    coefficients = cho_solve(cho_factor(gram+GAMMA*np.eye(1000), lower=True), target)
    beta2 = float(coefficients@gram@coefficients)
    c2 = float(np.sum(q))
    return {"quadrature_points": points, "beta_squared": beta2,
            "c_squared_quadrature": c2,
            "squared_bound_continuous_mass": beta2*(B+A/20),
            "squared_bound_quadrature_mass": beta2*(B+A*c2)}


def save_verified_figure(time, u, y, response, output_dir):
    # Preserve the original figure's 336 x 240 point canvas.
    fig, ax = plt.subplots(figsize=(336/72, 240/72), layout="constrained")
    ax.plot(time, .1*u, label="input (scaled by 0.1)", color="tab:blue")
    ax.plot(time, y, label="original system", color="tab:orange")
    ax.plot(time, response, label="approximation", color="tab:green")
    ax.set_xlabel("time/s")
    ax.set_ylabel("output")
    ax.set_xlim(0, 2)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.savefig(output_dir/"verified_example4_response.pdf")
    fig.savefig(output_dir/"verified_example4_response.png", dpi=200)
    plt.close(fig)


@numba.njit
def feedback_response(histories, coefficients, weight, point_weight, u):
    """Port the notebook's scattered feedback, with consistent input scaling."""
    count, width = histories.shape
    q = SCALE**2*A*DT*weight**2
    endpoint_q = q[-1] + SCALE**2*B*point_weight**2
    norm_past = np.zeros(count)
    for i in range(count):
        for k in range(width-1):
            norm_past[i] += q[k]*histories[i, k]**2
    z = np.zeros(len(u))
    output = np.zeros(len(u))
    residuals = np.zeros(len(u))
    iterations = np.zeros(len(u), dtype=np.int64)
    loop = 0.0
    past_distance = np.empty(count)
    for n in range(len(u)):
        start = width-1-n
        query_norm = 0.0
        for j in range(n):
            query_norm += q[start+j]*z[j]**2
        for i in range(count):
            cross = 0.0
            for j in range(n):
                cross += q[start+j]*histories[i, start+j]*z[j]
            past_distance[i] = max(norm_past[i] + query_norm-2*cross, 0.0)
        for iteration in range(10000):
            argument = u[n]-loop
            value = 0.0
            for i in range(count):
                distance = past_distance[i]+endpoint_q*(histories[i, -1]-argument)**2
                value += coefficients[i]*np.exp(-distance/(2*SIGMA**2))/SCALE
            residual = abs(value-loop)
            loop = value
            if residual < 1e-8:
                break
        iterations[n] = iteration+1
        residuals[n] = residual
        z[n] = u[n]-loop
        output[n] = u[n]-2*loop
    return output, residuals, iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output/example4_validation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("printed_lambda_10_analytic", "analytic", 10.0, False, False),
        ("code_exponent_10_analytic", "analytic", 20.0, False, False),
        ("code_exponent_10_euler", "euler", 20.0, False, False),
        ("normalized_unweighted_point_analytic", "analytic", 20.0, True, False),
        ("normalized_weighted_point_analytic", "analytic", 20.0, True, True),
    ]
    results = []
    selected = {}
    for case in cases:
        result, histories, coefficients, weight, point_weight = fit_case(*case)
        results.append(result)
        if case[0] in ("printed_lambda_10_analytic", "code_exponent_10_analytic",
                       "code_exponent_10_euler"):
            selected[case[0]] = (histories, coefficients, weight, point_weight)
        print(case[0], "beta2=", result["beta_squared"],
              "square (continuous c)=", result["squared_bound_continuous_mass"],
              "square (discrete c)=", result["squared_bound_discrete_mass"], flush=True)

    u, y, x = plant()
    _, _, rk = plant("rk4")
    rk_error = float(np.max(abs(x-rk)))
    assert rk_error < 1e-6
    recovered = {}
    for name, (histories, coefficients, weight, point_weight) in selected.items():
        response, residuals, iterations = feedback_response(histories, coefficients,
                                                            weight, point_weight, u[-1])
        assert iterations.max() < 10000
        assert residuals.max() < 1e-8
        recovered[name] = response
        result = next(r for r in results if r["name"] == name)
        result.update(feedback_max_iterations=int(iterations.max()),
                      feedback_max_residual=float(residuals.max()),
                      feedback_min=float(response.min()), feedback_max=float(response.max()))
        print(name, "feedback range",response.min(),response.max(),flush=True)

    reference_file = ROOT/"Kernel_work2/reference/example4_curves.npz"
    figure_comparison = {}
    reference = np.load(reference_file) if reference_file.exists() else None
    if reference is not None:
        for name, response in recovered.items():
            figure_comparison[name] = float(np.sqrt(np.mean((response-reference["green"])**2)))
        figure_comparison["analytic_original_orange_rmse"] = float(np.sqrt(np.mean((y[-1]-reference["orange"])**2)))
        figure_comparison["euler_original_orange_rmse"] = float(np.sqrt(np.mean((plant("euler")[1][-1]-reference["orange"])**2)))
        print("Original figure comparison", json.dumps(figure_comparison),flush=True)

    refinement = [refined_continuous_history(points) for points in [1000, 2000, 4000]]
    for row in refinement:
        print("History quadrature refinement", json.dumps(row), flush=True)
        assert row["squared_bound_quadrature_mass"] < 0.998
    assert abs(refinement[-1]["squared_bound_quadrature_mass"]
               - refinement[-2]["squared_bound_quadrature_mass"]) < 1e-5
    actual = next(r for r in results if r["name"] == "code_exponent_10_analytic")
    printed = next(r for r in results if r["name"] == "printed_lambda_10_analytic")
    assert actual["squared_bound_discrete_mass"] < 0.998
    assert printed["squared_bound_continuous_mass"] > 1

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
    axes[0].plot(TIME, y[-1], label="Original saturated lag", color="tab:orange")
    for name, response in recovered.items():
        axes[0].plot(TIME, response, label=name.replace("_", " "), alpha=.85)
    if reference is not None:
        axes[0].plot(reference["t"], reference["green"], "k--", label="Existing figure: passive approximation")
        for name, response in recovered.items():
            axes[1].plot(TIME, response-reference["green"], label=name.replace("_", " "))
    axes[0].set_ylabel("Output")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Difference from existing figure")
    axes[1].set_xlabel("Time (s)")
    axes[1].axhline(0, color="0.6", linewidth=.7)
    for ax in axes: ax.grid(alpha=.25)
    fig.savefig(args.output_dir/"reconstruction_comparison.png", dpi=180)
    plt.close(fig)
    # The official plot is generated from the same fit used for the certificate.
    save_verified_figure(TIME, u[-1], y[-1], recovered["code_exponent_10_analytic"], args.output_dir)
    np.savez_compressed(args.output_dir/"recovered_responses.npz", time=TIME,
                        input=u[-1], original=y[-1], **recovered)
    report = {
        "parameters": {"N": N, "duration": T, "omega_rad_s": 10,
                       "amplitudes": AMPLITUDES.tolist(), "scale": SCALE,
                       "gamma": GAMMA, "a": A, "b": B, "sigma": SIGMA,
                       "history_endpoints_one_based": (ENDPOINTS+1).tolist(),
                       "quadrature_dt": DT, "sample_grid_dt": float(TIME[1]-TIME[0])},
        "cases": results, "rk4_max_state_error": rk_error,
        "continuous_history_refinement": refinement,
        "original_figure_rmse": figure_comparison,
        "scope": "Reconstruction using the thesis's saturated lag and parameter recipe, and the saved code's scatter/kernel/history operations. The exact historical notebook state is not available.",
        "scaling": "If E_phys(u)=E_scaled(s*u)/s, equal input/output scaling cancels from the physical Lipschitz constant.",
        "normalization": "If histories including their last sample are multiplied by w, the endpoint contribution is b*w(0)^2. The notebook's printed diagnostic omits this endpoint factor when w is normalized.",
        "discretization": "The exact finite-delay quadrature certificate uses sum(w^2)*dt, which can exceed the continuous integral for an inclusive rectangular grid.",
        "conclusion": "For the unnormalized code weight exp(10*tau), the thesis parameter must be lambda=20 under w=exp(lambda*tau/2). The reproduced squared certificate is below 0.998, but 0.998 is a conservative bound rather than an exact equality. The printed lambda=10 convention does not pass this sufficient test at the stated gamma.",
    }
    (args.output_dir/"results.json").write_text(json.dumps(report, indent=2)+"\n",encoding="utf-8")
    lines = ["Example 4 numerical audit", "", report["scope"], "",
             "The table distinguishes the squared gain certificate from its square root.",
             "Case | beta^2 | squared bound (continuous mass) | gain bound | squared bound (discrete mass)"]
    for row in results:
        lines.append(f"{row['name']} | {row['beta_squared']:.12f} | "
                     f"{row['squared_bound_continuous_mass']:.12f} | "
                     f"{row['gain_bound_continuous_mass']:.12f} | "
                     f"{row['squared_bound_discrete_mass']:.12f}")
    lines += ["", f"Independent RK4/analytic maximum state discrepancy: {rk_error:.3e}",
              "Original-figure RMSE comparisons:", json.dumps(figure_comparison, indent=2),
              "", "Continuous-history quadrature refinement:", json.dumps(refinement, indent=2),
              "", report["scaling"], report["normalization"], report["discretization"],
              "", "Conclusion:", report["conclusion"],
              "The official reconstruction plot is verified_example4_response.pdf (and .png)."]
    (args.output_dir/"report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
