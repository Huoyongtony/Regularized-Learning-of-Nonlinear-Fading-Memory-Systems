# Kernel Modeling of Fading Memory Systems

Yongkang Huo, Thomas Chaffey, and Rodolphe Sepulchre introduce a kernel method for learning how a system’s past input determines its present output. Their paper demonstrates the method on linear and nonlinear systems. Chapter 4 of Huo’s dissertation **expands the paper**. It gives the sampling settings for those examples, revisits the passivity experiments, and develops gain-scheduling kernels to model potassium and sodium currents in a Hodgkin–Huxley neuron.

| Where to start | What you’ll find |
| --- | --- |
| [Paper and original notebooks](paper/README.md) | Section IV’s four examples, the linear and nonlinear Julia notebooks, and the corresponding chapter figures. |
| [Chapter 4](chapter/chapter.pdf) · [Chapter guide](chapter/README.md) | The expanded treatment, all 14 figures, and the LaTeX source and build instructions. |
| [Reproduce the chapter experiments](chapter/experiments/README.md) | Commands for the revised Examples 2 and 4, reference results, and the closed-loop Hodgkin–Huxley test. The [saved current cited on page 50](Kernel_work/code/test2_input.txt) contains the original 4,000 samples. |

## Cite the paper

Yongkang Huo, Thomas Chaffey, and Rodolphe Sepulchre, “Kernel Modeling of Fading Memory Systems,” *IEEE Transactions on Automatic Control*, 71(3), 2046–2052, March 2026. [DOI](https://doi.org/10.1109/TAC.2025.3621991) · [IEEE PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11204380) · [Open preprint](https://arxiv.org/abs/2403.11945) · [BibTeX](CITATION.bib) · [CITATION.cff](CITATION.cff)
