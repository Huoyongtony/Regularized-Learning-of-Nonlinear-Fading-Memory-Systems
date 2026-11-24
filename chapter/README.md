# Chapter 4 · Kernel Modelling and Identification of Ion Channels

Chapter 4 expands the paper’s kernel model of fading memory. Sections 4.1–4.2 develop the framework and introduce gain-scheduling kernels for memristive elements. Section 4.3 revisits the four paper examples with explicit sampling settings, then applies the new kernels to ion channels. [Read the chapter](chapter.pdf) · [Paper examples and corresponding chapter figures](../paper/README.md)

## Find a figure

| Chapter figures | Experiment | Starting point |
| --- | --- | --- |
| 4.1 | Scattering transform | [Diagram](figures/03_Kernel_Modelling_and_Identification_of_Neuromorphic_Elements/codes/scattering.png) |
| 4.2–4.3 | Example 1 · Fading weights and regularization | [Original linear notebook](<../kernel regression (linear) paper.ipynb>) |
| 4.4 | Example 2 · Sinusoidal training for the passive linear fit | [Chapter script](../Kernel_work2/codes/reproduce_example2_sinusoids.py) |
| 4.5 | Example 3 · Nonlinear RBF approximation | [Original nonlinear notebook](<../kernel regression (non-linear) paper-final-cleanup.ipynb>) |
| 4.6 | Example 4 · Feedback response and squared gain bound | [Chapter reconstruction](../Kernel_work2/codes/validate_example4.py) |
| 4.7–4.10 | RBF and gain-scheduling kernels trained on steps or sinusoids | [Julia experiment notebook](../Kernel_work/code/kernel_learning_new.ipynb) |
| 4.11–4.12 | Potassium and sodium open-loop tests | [Channel fitting and open-loop cells](../Kernel_work/code/kernel_learning_new.ipynb) |
| 4.13–4.14 | Closed-loop Hodgkin–Huxley voltage and injected current | [Julia runner](../Kernel_work/code/reproduce_hh_saved_input.jl) · [Saved current](../Kernel_work/code/test2_input.txt) |

For the Python and Julia commands, expected outputs, and reference reports, use the [chapter experiment guide](experiments/README.md).

## Build the standalone chapter

The [chapter body](chapter-content.tex) and [14 figures](figures/) come from the approved dissertation. The [standalone wrapper](chapter.tex) uses the dissertation class and mathematical style. It retains the chapter and figure numbers and starts on printed page 25. A [context page](context.tex) supplies equation (3.3) from Chapter 3. The standalone bibliography numbers the [19 cited references](references.bib) within this edition. The [provenance file](provenance.json) records source and figure checksums.

With a complete TeX Live installation, run these commands from `chapter/`. The supplied [formatted bibliography](chapter.bbl) supports a direct build.

```sh
pdflatex -interaction=nonstopmode -halt-on-error chapter.tex
pdflatex -interaction=nonstopmode -halt-on-error chapter.tex
```

After editing citations or bibliography entries, run `pdflatex chapter.tex`, then `biber chapter`, followed by two more pdfLaTeX passes.
