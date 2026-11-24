# The paper and its original examples

*Kernel Modeling of Fading Memory Systems* fits a kernel model to pairs of input histories and present outputs. Section IV illustrates how the fading-memory weight and regularization affect the fit, and how to enforce incremental passivity on linear and nonlinear examples. [IEEE article](https://ieeexplore.ieee.org/document/11204380) · [Section IV in the open preprint](https://arxiv.org/html/2403.11945v4#S4)

Start with the [linear Julia notebook](<../kernel regression (linear) paper.ipynb>) for the linear kernel experiments, or the [nonlinear Julia notebook](<../kernel regression (non-linear) paper-final-cleanup.ipynb>) for the saturated-system experiments. Open the notebooks in Jupyter with an IJulia kernel. Both record Julia 1.8.2 and list their packages in the first code cell. Before exporting figures, set the notebook’s `Folder` variable to a writable output directory with a trailing path separator.

## Where the examples appear in Chapter 4

| Paper example | Paper figures | Chapter figures | How the chapter develops the example |
| --- | --- | --- | --- |
| 1. Fading weight and regularization in a linear system | 2–3 | 4.2–4.3 | Specifies the training frequencies, amplitudes, input histories, and sampling grid. |
| 2. Passive approximation of a linear system | 4 | 4.4 | Revisits the fit using 100 sinusoidal inputs. The paper describes an impulse and 100 sinusoids. |
| 3. RBF approximation of a saturated first-order lag | 5 | 4.5 | Gives the training and sampling settings for the step-response comparison. |
| 4. Incrementally passive approximation of the saturated lag | 6 | 4.6 | Reconstructs the feedback response and checks the squared gain bound using the chapter’s explicit weight convention. |

The chapter then introduces gain-scheduling kernels, compares their generalization to the RBF kernel, and fits two ion-channel currents for open-loop and closed-loop tests (Figures 4.7–4.14). [Read Chapter 4](../chapter/chapter.pdf) · [Chapter experiments and reference results](../chapter/experiments/README.md)
