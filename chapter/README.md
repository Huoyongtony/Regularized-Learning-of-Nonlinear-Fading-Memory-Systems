# Chapter 4: Kernel Modelling and Identification of Ion Channels

[Read the chapter](chapter.pdf). This standalone edition contains the complete
chapter body from the approved dissertation source, including its definitions,
proofs, experiments, sampling tables, and all 14 figures. The chapter number
remains 4 and page numbering starts at 25. References are numbered for this
standalone edition and may differ from the full dissertation. A short context
page supplies equation (3.3), which the chapter cites from Chapter 3.

## Source and build

- `chapter-content.tex`: chapter body copied from the dissertation source.
- `chapter.tex`: standalone wrapper using the dissertation class and preamble.
- `context.tex`: context for the references to Chapter 3 and its HH equation.
- `figures/`: all figures included by the chapter, copied from the dissertation.
- `references.bib`: the 19 bibliography entries cited in the chapter.
- `chapter.bbl`: preformatted bibliography, supplied to support direct builds.
- `Classes/` and `sty/`: original template and mathematical style dependencies.
- `provenance.json`: chapter-source and figure checksums and bibliography keys.

With a complete TeX Live installation, run from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error chapter.tex
pdflatex -interaction=nonstopmode -halt-on-error chapter.tex
```

The supplied `chapter.bbl` supports these direct builds. If you change citations
or bibliography entries, regenerate it with Biber between LaTeX passes:

```sh
pdflatex chapter.tex
biber chapter
pdflatex chapter.tex
pdflatex chapter.tex
```

The template class carries its original license and attribution in its source.
The generated chapter has been compiled and checked for unresolved references
and overflowing lines. The dissertation workspace is not needed to rebuild it.

## Figure-to-code guide

Paths below are relative to the repository root. Install dependencies and use
the run commands in the [repository README](../README.md).

| Figures | Content | Code or source |
| --- | --- | --- |
| 4.1 | Scattering-transform diagram | Static diagram supplied as `chapter/figures/03_Kernel_Modelling_and_Identification_of_Neuromorphic_Elements/codes/scattering.png` |
| 4.2-4.3 | Linear kernel: fading weights and regularization | `kernel regression (linear) paper.ipynb` |
| 4.4 | Corrected sinusoidal-only linear passivity example | `Kernel_work2/codes/reproduce_example2_sinusoids.py` |
| 4.5 | Nonlinear RBF approximation | `kernel regression (non-linear) paper-final-cleanup.ipynb` |
| 4.6 | Corrected nonlinear passive reconstruction | `Kernel_work2/codes/validate_example4.py` |
| 4.7-4.10 | GS/RBF step-versus-sinusoid comparisons | Historical experiments in `Kernel_work/code/kernel_learning_new.ipynb`; the chapter tables specify the approved settings |
| 4.11-4.12 | Potassium and sodium open-loop tests | Channel fitting and open-loop cells in `Kernel_work/code/kernel_learning_new.ipynb`, with the three adjacent `.jl` modules |
| 4.13-4.14 | HH closed-loop voltage and saved injected current | `Kernel_work/code/reproduce_hh_saved_input.jl` and `Kernel_work/code/test2_input.txt` |

The historical notebooks contain exploratory alternatives and require selecting
the relevant cells and settings; they are not automated builds of every final
figure. The ordered HH runner and corrected Python scripts provide explicit
entry points for those experiments. Julia execution has not been verified in
this update environment. The Python scripts for Figures 4.4 and 4.6 were run
successfully, including their numerical assertions.

The chapter supplies the experimental specification; preserving historical
source does not establish bit-for-bit recovery of every original execution.
In particular, the original HH random seed was not recorded, and Example 4
uses the documented reconstruction adopted for the corrected figure.
