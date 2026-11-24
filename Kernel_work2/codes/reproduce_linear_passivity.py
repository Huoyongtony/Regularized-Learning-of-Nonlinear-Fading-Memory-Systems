"""Reproduce the linear incremental-passivity example in Section 4.3.

The experiment learns the scattering transform

    E = (I - G) (I + G)^(-1)

of G(s) = 10 / ((s + 1)(s + 2)) with the modified bilinear kernel.
The learned passive approximation is recovered as

    G_hat = (I - E_hat) (I + E_hat)^(-1).

Only NumPy and ReportLab are required. This is a legacy reconstruction;
reproduce_example2_sinusoids.py implements the approved sinusoidal-only example.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.pdfgen import canvas


def original_response(omega: float, t: np.ndarray) -> np.ndarray:
    """Zero-state response of 10/((s+1)(s+2)) to sin(omega*t)."""

    def first_order_convolution(rate: float) -> np.ndarray:
        return (
            rate * np.sin(omega * t)
            - omega * np.cos(omega * t)
            + omega * np.exp(-rate * t)
        ) / (rate * rate + omega * omega)

    return 10.0 * (first_order_convolution(1.0) - first_order_convolution(2.0))


def trapezoid_weights(size: int, dt: float) -> np.ndarray:
    weights = np.full(size, dt, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def learn_scattered_functional(
    dt: float,
    include_endpoint_impulse: bool,
    fading_exponent_multiplier: float,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    # Parameters stated in the thesis.
    duration = 30.0
    history = 4.0
    fading_rate = 0.5
    a = 1.0
    b = 1.0

    frequencies = np.exp(np.linspace(-1.0, 3.0, 100))
    tau = np.arange(-history, 0.5 * dt, dt)
    absolute_time = duration + tau

    transformed_histories: list[np.ndarray] = []
    transformed_targets: list[float] = []
    for omega in frequencies:
        u = np.sin(omega * absolute_time)
        y = original_response(omega, absolute_time)
        transformed_histories.append(0.5 * (u + y))
        transformed_targets.append(float(0.5 * (u[-1] - y[-1])))

    if include_endpoint_impulse:
        # The thesis does not state the impulse time or numerical convention.
        # A unit endpoint sample is the least committal discrete interpretation:
        # the strictly proper plant has zero simultaneous output, and this sample
        # identifies the instantaneous component introduced by scattering.
        impulse_u = np.zeros_like(tau)
        impulse_u[-1] = 1.0
        transformed_histories.append(0.5 * impulse_u)
        transformed_targets.append(0.5)

    x_train = np.asarray(transformed_histories)
    y_train = np.asarray(transformed_targets)

    quadrature = trapezoid_weights(tau.size, dt)
    fading_density = np.exp(fading_exponent_multiplier * fading_rate * tau)
    integral_weights = quadrature * fading_density

    gram = a * (x_train * integral_weights) @ x_train.T
    gram += b * np.outer(x_train[:, -1], x_train[:, -1])
    coefficients = np.linalg.solve(
        gram + regularization * np.eye(gram.shape[0]), y_train
    )

    representer = coefficients @ x_train
    rkhs_norm = float(np.sqrt(coefficients @ gram @ coefficients))
    c_squared_truncated = float(np.sum(integral_weights))
    c_squared_full = 1.0 / (fading_exponent_multiplier * fading_rate)
    certified_gain_truncated = rkhs_norm * math.sqrt(
        b + a * c_squared_truncated
    )
    certified_gain_full = rkhs_norm * math.sqrt(b + a * c_squared_full)

    diagnostics = {
        "rkhs_norm": rkhs_norm,
        "certified_gain_full_history": certified_gain_full,
        "certified_gain_truncated_history": certified_gain_truncated,
        "training_points": float(x_train.shape[0]),
        "history_samples": float(tau.size),
    }
    return frequencies, tau, integral_weights, representer, diagnostics


def frequency_responses(
    frequencies: np.ndarray,
    tau: np.ndarray,
    integral_weights: np.ndarray,
    representer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.exp(1j * np.outer(frequencies, tau))
    e_hat = phase @ (representer * integral_weights)
    e_hat += representer[-1]
    g_hat = (1.0 - e_hat) / (1.0 + e_hat)
    g_true = 10.0 / ((1j * frequencies + 1.0) * (1j * frequencies + 2.0))
    return g_true, e_hat, g_hat


def draw_nyquist_pdf(
    output: Path,
    g_true: np.ndarray,
    g_hat: np.ndarray,
    diagnostics: dict[str, float],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720.0, 540.0
    c = canvas.Canvas(str(output), pagesize=(width, height))

    left, right = 78.0, 28.0
    bottom, top = 64.0, 62.0
    plot_w = width - left - right
    plot_h = height - bottom - top
    x_min, x_max = -0.7, 4.2
    y_min, y_max = -3.25, 0.1

    def xp(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def yp(y: float) -> float:
        return bottom + (y - y_min) / (y_max - y_min) * plot_h

    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFont("Helvetica", 30)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2.0, height - 38.0, "Nyquist plot")

    c.setStrokeColor(colors.HexColor("#d7d7d7"))
    c.setLineWidth(0.6)
    for x in (-0.5, 0.0, 1.0, 2.0, 3.0, 4.0):
        c.line(xp(x), bottom, xp(x), bottom + plot_h)
    for y in (0.0, -1.0, -2.0, -3.0):
        c.line(left, yp(y), left + plot_w, yp(y))

    c.setStrokeColor(colors.black)
    c.setLineWidth(1.4)
    c.line(left, bottom, left, bottom + plot_h)
    c.line(left, bottom, left + plot_w, bottom)

    c.setFont("Helvetica", 13)
    for x in (0.0, 1.0, 2.0, 3.0, 4.0):
        c.drawCentredString(xp(x), bottom - 21.0, f"{x:g}")
    for y in (0.0, -1.0, -2.0, -3.0):
        c.drawRightString(left - 10.0, yp(y) - 4.0, f"{y:g}")

    c.setFont("Helvetica", 18)
    c.drawCentredString(left + plot_w / 2.0, 20.0, "Real")
    c.saveState()
    c.translate(20.0, bottom + plot_h / 2.0)
    c.rotate(90)
    c.drawCentredString(0.0, 0.0, "Imaginary")
    c.restoreState()

    true_color = colors.HexColor("#e86f43")
    learned_color = colors.HexColor("#149ee7")
    c.setLineWidth(0.6)
    for response, color in ((g_hat, learned_color), (g_true, true_color)):
        c.setFillColor(color)
        c.setStrokeColor(colors.black)
        for value in response:
            x = float(value.real)
            y = float(value.imag)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                c.circle(xp(x), yp(y), 4.2, fill=1, stroke=1)

    legend_x, legend_y = 365.0, 423.0
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.0)
    c.rect(legend_x, legend_y - 61.0, 306.0, 61.0, fill=1, stroke=1)
    c.setFont("Helvetica", 14)
    c.setFillColor(learned_color)
    c.circle(legend_x + 22.0, legend_y - 20.0, 5.0, fill=1, stroke=1)
    c.setFillColor(colors.black)
    c.drawString(legend_x + 39.0, legend_y - 25.0, "Passive approximation")
    c.setFillColor(true_color)
    c.circle(legend_x + 22.0, legend_y - 45.0, 5.0, fill=1, stroke=1)
    c.setFillColor(colors.black)
    c.drawString(legend_x + 39.0, legend_y - 50.0, "Real system")

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#444444"))
    note = (
        f"Reconstructed from 100 sine responses and one endpoint impulse. "
        f"Certified scattered gain {diagnostics['certified_gain_full_history']:.3f}."
    )
    c.drawRightString(width - 28.0, 7.0, note)
    c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--gamma", type=float, default=14.8)
    parser.add_argument(
        "--fading-exponent-multiplier",
        type=float,
        default=2.0,
        help=(
            "2 reproduces the older notebooks, which multiply signals by "
            "exp(lambda*tau) before taking their Euclidean inner product. "
            "Use 1 to follow the inner-product density printed in the thesis."
        ),
    )
    parser.add_argument(
        "--no-impulse",
        action="store_true",
        help="omit the endpoint impulse used to identify direct feedthrough",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/reproduced_linear_passivity.pdf"),
    )
    args = parser.parse_args()

    frequencies, tau, weights, representer, diagnostics = learn_scattered_functional(
        args.dt,
        not args.no_impulse,
        args.fading_exponent_multiplier,
        args.gamma,
    )
    g_true, e_hat, g_hat = frequency_responses(
        frequencies, tau, weights, representer
    )

    diagnostics.update(
        {
            "minimum_real_part": float(np.min(g_hat.real)),
            "maximum_scattered_gain": float(np.max(np.abs(e_hat))),
            "minimum_true_real_part": float(np.min(g_true.real)),
        }
    )
    draw_nyquist_pdf(args.output, g_true, g_hat, diagnostics)

    for key, value in diagnostics.items():
        if key.endswith("points") or key.endswith("samples"):
            print(f"{key}={int(value)}")
        else:
            print(f"{key}={value:.9f}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
