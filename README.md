# Kernel Modeling of Fading Memory Systems

Companion code for the paper **Kernel Modeling of Fading Memory Systems** by
Yongkang Huo, Thomas Chaffey, and Rodolphe Sepulchre, and for the related
dissertation chapter **Kernel Modelling and Identification of Ion Channels**.

**Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11204380) ·
[Published PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11204380) ·
[DOI](https://doi.org/10.1109/TAC.2025.3621991) ·
[Open preprint](https://arxiv.org/abs/2403.11945)

The original paper notebooks are
[`kernel regression (linear) paper.ipynb`](<kernel regression (linear) paper.ipynb>)
and
[`kernel regression (non-linear) paper-final-cleanup.ipynb`](<kernel regression (non-linear) paper-final-cleanup.ipynb>).

The dissertation materials extend the presentation with gain-scheduling kernels,
ion-channel and HH experiments, detailed sampling tables, and reproduction
scripts for the approved chapter corrections. Read the complete chapter in
[chapter/chapter.pdf](chapter/chapter.pdf), with its
[LaTeX source, figures, bibliography, build instructions, and figure-to-code guide](chapter/README.md).
The saved HH current cited on dissertation page 50 is
[`Kernel_work/code/test2_input.txt`](Kernel_work/code/test2_input.txt).
Example numbers and correction labels below refer to the dissertation chapter.

## Cite the paper

Yongkang Huo, Thomas Chaffey, and Rodolphe Sepulchre, "Kernel Modeling of Fading
Memory Systems," *IEEE Transactions on Automatic Control*, vol. 71, no. 3,
pp. 2046–2052, March 2026. DOI:
[10.1109/TAC.2025.3621991](https://doi.org/10.1109/TAC.2025.3621991).

Citation files: [BibTeX](CITATION.bib) · [CITATION.cff](CITATION.cff).

## Contents

| Experiment | Entry point |
| --- | --- |
| Complete chapter text, settings tables, figures, and references | [`chapter/chapter.pdf`](chapter/chapter.pdf) |
| Paper: original linear examples | `kernel regression (linear) paper.ipynb` |
| Paper: original nonlinear examples | `kernel regression (non-linear) paper-final-cleanup.ipynb` |
| Gain-scheduling, ion channels, and HH exploratory work | `Kernel_work/code/kernel_learning_new.ipynb` |
| HH closed-loop test with the saved current | `Kernel_work/code/reproduce_hh_saved_input.jl` |
| Approved Example 2: sinusoidal-only passive linear fit | `Kernel_work2/codes/reproduce_example2_sinusoids.py` |
| Approved Example 4: nonlinear passive fit and numerical audit | `Kernel_work2/codes/validate_example4.py` |

The two original root notebooks are retained unchanged. The added historical
notebook has its source cells preserved, with embedded outputs and execution
counts cleared to avoid a 90 MB file. It contains exploratory cells and depends
on interactive execution order; use the HH script for the ordered closed-loop
experiment. The corrected Python scripts take precedence over historical
notebook settings for dissertation Examples 2 and 4.

## Saved HH current (page 50)

[`Kernel_work/code/test2_input.txt`](Kernel_work/code/test2_input.txt) is copied
byte for byte from the dissertation materials. It contains 4,000 finite values,
one current sample per line. Its SHA-256 is:

```text
9d957ba9bd87449479cd5f281a66efd0a2688eaef3798aadd7136d6f7b2e4a31
```

The historical generator was

```text
I[i] = 3 + sum(xi[i:i+500]) / 100, i = 1,...,4000
xi[1:4500] ~ independent Uniform(-50, 50)
```

Each sum has 501 terms. The seed for this saved realization was not recorded;
use the supplied vector to reproduce the experiment. A new draw, including one
made after a notebook's seed initialization, does not specify the same input.
Both models use current samples 1 through 3999 for forward-Euler updates with
step 0.01. The last sample appears in the input plot only. The historical plots
place all 4,000 samples on an inclusive 0-to-40 axis. The original HH trace
records the voltage before each update, whereas the learned trace records it
after each update; the runner preserves this historical plotting convention.

### Julia setup and HH run

The historical channel notebook used Julia 1.9.3. The root notebooks recorded
Julia 1.8.2. Original package manifests are unavailable, so package-version
equivalence to the historical environment cannot be guaranteed.

From the repository root, install the HH runner's dependencies into a local
Julia environment:

```sh
julia --project=Kernel_work/code -e 'using Pkg; Pkg.add(["ControlSystems", "ControlSystemsBase", "OrdinaryDiffEq", "TensorCast", "Plots"])'
julia --project=Kernel_work/code --threads=auto Kernel_work/code/reproduce_hh_saved_input.jl
```

The runner uses the original model, potassium and sodium training cells, and
closed-loop cells, arranged in dependency order. It reads the saved input and
writes figures and `closed_loop_traces.txt` to `output/hh_saved_input/`. The trace
columns are plotting time, injected current, original voltage, and learned
voltage. An optional first argument selects another output directory. Training
uses 1,000 potassium pairs and 2,000 sodium pairs and can be expensive. This
runner has not been executed in the update environment because Julia is not
installed there.

For the historical notebooks, install Jupyter and IJulia, select a Julia kernel,
and run from the notebook's directory. Additional imports are listed in their
first cells (including Revise, ForwardDiff, Distributions, Convex, SCS, and Optim
in `kernel_learning_new.ipynb`). The three local `.jl` modules are supplied in
the same directory.

## Approved corrections: Python reproduction

Use Python 3.12 and install the tested dependencies:

```sh
python -m pip install -r requirements.txt
python Kernel_work2/codes/reproduce_example2_sinusoids.py
python Kernel_work2/codes/validate_example4.py
```

Both scripts run numerical assertions and write reports, fitted responses, and
figures under `output/`. Use `--output-dir PATH` to select another destination.
The reports retained from the dissertation correction work are in
`reference_results/` for comparison.

### Example 2

The approved example uses 100 zero-phase, unit-amplitude sinusoidal inputs with
angular frequencies `exp(linspace(-1, 3, 100))`, simulated from zero state for
30 seconds. Each training history covers the final 4 seconds. The baseline
uses step 0.002, trapezoidal quadrature, `w(tau)=exp(tau/2)`, `lambda=1`,
`a=b=1`, and `gamma=14.8`.

The script independently checks RK4 against the analytic plant response,
compares two fitting implementations, checks six sampling grids, and compares
against exact history integration. The resulting gain bound is approximately
0.993370836164, conservatively reported as 0.994. The generated Nyquist plot is
`output/example2_sinusoids_only/scatter_linear_dots.pdf`.

### Example 4

The script reconstructs the saturated first-order lag, scattering data,
modified RBF kernel, regularized fit, and recovered feedback response. It
checks analytic and RK4 plant responses, feedback convergence, and refined
history quadrature. The corrected weight is `w(tau)=exp(10*tau)`, corresponding
to `lambda=20` under `w(tau)=exp(lambda*tau/2)`; the reported squared gain bound
is below 0.998. The generated approved figure is
`output/example4_validation/verified_example4_response.pdf`.

The saved historical notebook execution state was not recovered. This is the
explicit reconstruction used for the corrected figure. The original figure
and its digitized paths are supplied in `Kernel_work2/reference/`; comparison
with those paths has finite PDF-coordinate precision.

### Optional historical checks

```sh
python -m pip install -r requirements-optional.txt
python Kernel_work2/codes/validate_example2.py
python Kernel_work2/codes/extract_example4_reference.py
```

`validate_example2.py` and `reproduce_linear_passivity.py` retain the earlier
impulse/weight-convention investigations. Their defaults are historical audit
cases; the approved Example 2 is `reproduce_example2_sinusoids.py`.
`extract_example4_reference.py` regenerates the supplied paths from the included
original figure PDF.
