# Reproduce the Chapter 4 experiments

Chapter 4 revisits the paper’s linear and nonlinear passivity examples, then tests gain-scheduling kernels on ion-channel currents. Run the commands below from the **repository root**. The [chapter figure guide](../README.md) identifies the code for each figure, and the [paper guide](../../paper/README.md) maps the original examples to their chapter versions.

## Chapter versions of Examples 2 and 4

Use Python 3.12 and the [tested package versions](../../requirements.txt). Each script runs numerical checks and writes its figures, fitted responses, and report under `output/`. Both accept `--output-dir PATH`.

```sh
python -m pip install -r requirements.txt
python Kernel_work2/codes/reproduce_example2_sinusoids.py
python Kernel_work2/codes/validate_example4.py
```

| Figure | What to check | Expected output and reference |
| --- | --- | --- |
| 4.4 · Passive linear fit | With 100 unit-amplitude sinusoids, the gain bound is approximately 0.993370836164, reported as 0.994. | `output/example2_sinusoids_only/scatter_linear_dots.pdf` · [Reference report](reference_results/example2_sinusoids_only/report.txt) |
| 4.6 · Passive nonlinear fit | With the chapter’s weight `w(tau)=exp(10*tau)`, the squared gain bound is below 0.998. | `output/example4_validation/verified_example4_response.pdf` · [Reference report](reference_results/example4_validation/report.txt) |

**Example 2.** The sinusoidal angular frequencies are `exp(linspace(-1, 3, 100))`. Each simulation starts from zero and runs for 30 seconds. The fit uses the final four seconds, step 0.002, trapezoidal quadrature, `lambda=1`, `a=b=1`, and `gamma=14.8`. The script compares two fitting implementations, checks six sampling grids, and compares RK4 with the analytic response and the discrete fit with exact history integration. [Script](../../Kernel_work2/codes/reproduce_example2_sinusoids.py) · [Reference results](reference_results/example2_sinusoids_only/results.json)

**Example 4.** The script reconstructs the saturated first-order lag, scattering data, modified RBF kernel, and recovered feedback response. Under the chapter convention `w(tau)=exp(lambda*tau/2)`, the weight corresponds to `lambda=20`. The script checks feedback convergence and refines the history quadrature. The corrected figure uses this reconstruction. The original notebook execution state is unavailable, and comparisons with the digitized historical plot have finite PDF-coordinate precision. [Script](../../Kernel_work2/codes/validate_example4.py) · [Reference results](reference_results/example4_validation/results.json) · [Historical curves and PDF](../../Kernel_work2/reference/)

## Gain scheduling and ion channels

For Figures 4.7–4.12, open the [Julia experiment notebook](../../Kernel_work/code/kernel_learning_new.ipynb) in Jupyter with IJulia. The step and sinusoid experiments compare RBF and gain-scheduling kernels. The channel fits then use voltage steps for potassium and pulses for sodium. Use the settings tables in Section 4.3.4 when selecting exploratory cells. The notebook records Julia 1.9.3 and loads the three Julia modules in its directory. Its opening cells list the additional notebook dependencies. The archive preserves the source cells and omits the original embedded outputs.

### Closed-loop test with the saved current (Figures 4.13–4.14)

Use the exact current cited on dissertation page 50. Both models use samples 1–3999 for Euler updates of length 0.01 and plot the 4000th sample on the inclusive 0–40 time axis. The original HH trace records voltage before each update and the learned trace records it afterwards. [Saved 4,000-sample current](../../Kernel_work/code/test2_input.txt) · [Ordered Julia runner](../../Kernel_work/code/reproduce_hh_saved_input.jl)

```sh
julia --project=Kernel_work/code -e 'using Pkg; Pkg.add(["ControlSystems", "ControlSystemsBase", "OrdinaryDiffEq", "TensorCast", "Plots"])'
julia --project=Kernel_work/code --threads=auto Kernel_work/code/reproduce_hh_saved_input.jl
```

The runner trains on 1,000 potassium pairs and 2,000 sodium pairs. It writes the voltage and current plots and `closed_loop_traces.txt` under `output/hh_saved_input/`. The trace columns are plotting time, injected current, original voltage, and learned voltage. An optional first argument selects another output directory. The historical package manifest is unavailable. The two Python runs above were verified against the reference reports. The Julia runner still awaits execution in a Julia environment.

The historical generator drew 4,500 independent uniform samples on [−50, 50]. Each current value was 3 plus a sliding sum of 501 samples divided by 100. The seed for this realization was unrecorded. The saved file has SHA-256 `9d957ba9bd87449479cd5f281a66efd0a2688eaef3798aadd7136d6f7b2e4a31`.

## Optional historical comparisons

The Example 2 audit compares the impulse and weight conventions using an [earlier reconstruction](../../Kernel_work2/codes/reproduce_linear_passivity.py). The Example 4 extraction regenerates the supplied reference paths from the original figure PDF.

```sh
python -m pip install -r requirements-optional.txt
python Kernel_work2/codes/validate_example2.py
python Kernel_work2/codes/extract_example4_reference.py
```
