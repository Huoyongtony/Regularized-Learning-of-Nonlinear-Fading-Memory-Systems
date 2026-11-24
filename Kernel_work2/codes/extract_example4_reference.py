"""Extract the three original plot paths for a reproducible figure comparison.

Requires NumPy and pdfplumber. Coordinates in the source PDF are rounded, so
the resulting comparison has a small digitization error. It is not an exact
recovery of the original simulation arrays.
"""

import argparse
from pathlib import Path

import numpy as np
import pdfplumber


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path,
                        default=root/"Kernel_work2/reference/example4_original.pdf")
    parser.add_argument("--output", type=Path,
                        default=root/"Kernel_work2/reference/example4_curves.npz")
    args = parser.parse_args()
    with pdfplumber.open(args.pdf) as pdf:
        curves = [curve for curve in pdf.pages[0].curves
                  if len(curve.get("pts", [])) == 1000]
        assert len(curves) == 3, "Expected input, original, and approximation paths."
        points = [np.array(curve["pts"]) for curve in curves]
    t = 2*(points[0][:, 0]-points[0][0, 0])/(points[0][-1, 0]-points[0][0, 0])
    displayed_input = 15*np.sin(10*t)
    slope, offset = np.linalg.lstsq(
        np.column_stack((displayed_input, np.ones(len(t)))), points[0][:, 1], rcond=None)[0]
    signals = [(p[:, 1]-offset)/slope for p in points]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, t=t, blue=signals[0], orange=signals[1], green=signals[2])
    print(args.output)


if __name__ == "__main__":
    main()
