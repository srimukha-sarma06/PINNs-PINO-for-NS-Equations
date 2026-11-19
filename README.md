# PINNs & PINO for Navier–Stokes: Pipe Flow and Constriction Analysis

Physics-informed neural networks (PINNs) and a physics-informed neural operator (PINO, built on DeepONet) for **2D incompressible Navier–Stokes flow in a pipe**, developed for the **TRON programming contest 2026**.

The long-term goal is an **edge-AI system for fluid analysis in blood-vessel-like pipes**: reconstruct the pressure and velocity field from a few pressure-transducer readings, then use the pressure drop along the pipe to locate and size constrictions and estimate wall shear stress, all on a microcontroller with no cloud.

> **Scope of this repo.** The notebooks here cover (1) an *inverse PINN* that learns the flow field from sparse observations of a straight laminar channel, (2) a *PINO/DeepONet* that maps inlet/outlet pressure `(P_in, P_out)` to the full `(u, v, p)` field, and (3) export of the trained networks to TFLite for embedded targets. Constriction geometries and wall-shear-stress estimation are the intended downstream use and are **not yet demonstrated in these notebooks** (see [Roadmap](#roadmap)).

---

## Contents

- [Governing equations](#governing-equations)
- [Approaches](#approaches)
  - [A. Inverse PINN](#a-inverse-pinn-navier-stokesns_01_inverseipynb)
  - [B. PINO / DeepONet](#b-pino--deeponet-pino_tutorialpino_deepxdeipynb)
- [Results](#results)
- [Edge deployment](#edge-deployment)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Dataset](#dataset)
- [Limitations & known issues](#limitations--known-issues)
- [Roadmap](#roadmap)
- [References](#references)

---

## Governing equations

Unknowns are velocity $(u, v)$ and pressure $p$ as functions of $(x, y, t)$. For incompressible flow with density $\rho$ and kinematic viscosity $\nu$:

$$
\begin{aligned}
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} &= 0 \\
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial x} &= \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) \\
\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial y} &= \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)
\end{aligned}
$$

Continuity is enforced as an explicit PDE-loss term. The model works in the **laminar regime**, because PINNs/PINOs do not yet learn turbulent Navier–Stokes reliably. Everything is built with [DeepXDE](https://github.com/lululxvi/deepxde) on the TensorFlow backend.

---

## Approaches

### A. Inverse PINN (`Navier-Stokes/ns_01_inverse.ipynb`)

Learn the flow field **and** the viscosity from sparse observations of a 2D channel.

| Item | Setting |
|---|---|
| Domain | $x\in[0,1]$, $y\in[0,0.1]$, $t\in[0,0.1]$ |
| Network | FNN `[3] + [100]×4 + [3]`, `tanh`, Glorot-uniform; inputs $(x,y,t)$, outputs $(u,v,p)$ |
| Input scaling | feature transform multiplies $y$ and $t$ by 10 |
| Observations | 5,000 laminar samples of $(u, v)$ (**pressure is withheld** and inferred) |
| Unknown parameter | viscosity, as a trainable `dde.Variable` with $\nu=\theta^2/\rho$, $\rho=1$, $\theta_0=0.1$ |
| Boundary conditions | no-slip walls ($u=v=0$ at $y=0$ and $y=0.1$); outlet pressure $p=0$ at $x=1$ |
| Collocation points | 1,000 domain + 500 boundary, resampled every 1,000 steps (`PDEPointResampler`) |
| Optimiser | Adam (lr 1e-3, exponential decay 0.95 / 1,000 steps), 15k iterations with early stopping, then L-BFGS (max 30k iterations) |
| Loss weighting | boundary terms weighted ×5 relative to PDE/data terms |

`ns_02_inverse_cylindrical.ipynb` is an early exploration of the same problem in cylindrical geometry.

### B. PINO / DeepONet (`PINO_TUTORIAL/pino_deepxde.ipynb`)

Instead of fitting one flow, learn an **operator** from boundary pressures to the whole field, so a new `(P_in, P_out)` reading needs one forward pass and no retraining.

| Item | Setting |
|---|---|
| Domain | $x, y, t \in [0, 1]$ |
| Operator input (branch) | 2 sensors: $[P_{in}, P_{out}]$ |
| Operator coordinates (trunk) | $(x, y, t)$ |
| Function space | $P_{in}\sim\mathcal U(0.5, 2.0)$, $P_{out}\sim\mathcal U(0, 0.4)$, so $P_{in} > P_{out}$ always |
| Network | `dde.nn.DeepONet`, branch `[2, 96, 96]`, trunk `[3, 96, 96]`, `tanh`, 3 outputs (`split_both`) |
| Pressure BCs | **hard-constrained** via output transform: $p = (1-x)P_{in} + xP_{out} + x(1-x)\,\hat p$ |
| Other constraints | no-slip walls; initial condition $u=v=0$ at $t=0$ |
| Viscosity | fixed, $\nu = 0.1$ |
| Training | `PDEOperator`, 5 sampled functions per step, Adam (lr 1e-3, 15k iterations) then L-BFGS (15k) |

`PINO_TUTORIAL/pino_01.ipynb` is a from-scratch PyTorch DeepONet/PINO on a simple ODE, written to build intuition before moving to DeepXDE.

---

## Results

All numbers below are taken from the saved notebook outputs.

### A. Inverse PINN (straight channel)

| Metric | Value |
|---|---|
| Relative L2 error, $u$ (vs. observations) | 1.4 × 10⁻² |
| Mean PDE residual: continuity / x-momentum / y-momentum | 2.5 × 10⁻⁴ / 3.8 × 10⁻⁴ / 1.1 × 10⁻⁴ |
| Max \|v\| on the walls (no-slip check) | ≈ 1.7 × 10⁻⁴ (bottom), 1.1 × 10⁻⁴ (top) |
| Relative L2 error, $v$ and $p$ | ≈ 1.0 (see [Limitations](#limitations--known-issues)) |

### B. PINO / DeepONet

For $(P_{in}, P_{out}) = (2.0, 0.4)$ the network gives a linear pressure drop along the pipe and a developing, roughly parabolic velocity profile:

| Quantity | Value |
|---|---|
| $p(x=0.25,\ 0.50,\ 0.75)$ | 1.600, 1.201, 0.800 |
| Peak $u$ at $t = 0,\ 0.25,\ 0.5,\ 1.0$ | 0.034, 0.384, 0.730, 1.229 |
| Wall velocity at $t=1$ | ≈ 0.01 (no-slip satisfied to ~1%) |

<!-- TODO: add figures, e.g. ![u, v, p heatmaps](docs/figures/pinn_heatmaps.png), loss curves, PINO pressure profiles for several (Pin, Pout). -->

---

## Edge deployment

Trained networks are exported to **TFLite** with the scripts in `model-export/`:

| Model | File | Size |
|---|---|---|
| Inverse PINN (FNN) | `Navier-Stokes/models_2/pinn_model.tflite` | ≈ 127 KB |
| PINO, stream-function variant | `PINO_TUTORIAL/tflite_models/pino_model.tflite` | ≈ 18 KB |
| PINO, direct (u, v, p) variant | `PINO_TUTORIAL/tflite_models/pino_model_no_stream.tflite` | ≈ 31 KB |
| Custom Keras DeepONet (compact / deeper) | `PINO_TUTORIAL/tflite_models/custom_keras_model*.tflite` | ≈ 87 KB / 116 KB |

A DeepONet/PINN inverse variant (branch inputs $L, d, P_{in}, P_{out}$; trunk inputs $x, y, t$) was deployed on a **Renesas RA8P1** microcontroller and validated against ideal flow maps. *(The RA8P1 deployment code is not part of this repository.)*

<!-- TODO: add on-device latency / RAM / flash numbers and a photo or diagram of the setup. -->

---

## Repository structure

```
.
├── Navier-Stokes/            # Inverse PINN experiments (main work)
│   ├── ns_01_inverse.ipynb           # channel-flow inverse PINN
│   ├── ns_02_inverse_cylindrical.ipynb
│   ├── models/, models_2/            # saved weights (.pt / .weights.h5) and pinn_model.tflite
│   └── <run>/{loss,train,test}.dat   # DeepXDE training logs per run
├── PINO_TUTORIAL/            # PINO / DeepONet pipe-flow work
│   ├── pino_deepxde.ipynb            # DeepXDE PINO for (P_in, P_out) -> (u, v, p)
│   ├── pino_01.ipynb                 # from-scratch PyTorch DeepONet/PINO warm-up
│   └── tflite_models/, pino_models*/ # exported models and weights
├── data/
│   ├── navier-stokes/        # Flow.csv, Laminar_dataset.csv, Turbulent_dataset.csv
│   └── burgers/Burgers.npz
├── dataset-creation/         # dataset.ipynb: split Flow.csv into laminar / turbulent
├── model-export/             # Keras -> TFLite export scripts
├── pinns-pino-examples/      # Learning notebooks: Schrödinger, heat, Euler beam, ODEs, 2D
└── requirements.txt
```

---

## Getting started

**Tested with:** Python 3.11, DeepXDE 1.15.0, TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.

```bash
git clone https://github.com/srimukha-sarma06/PINNs-PINO-for-NS-Equations.git
cd PINNs-PINO-for-NS-Equations

python -m venv .venv && source .venv/bin/activate
pip install "deepxde==1.15.0" "tensorflow==2.15.1" "keras==2.15.0" \
            "numpy<2" pandas matplotlib scipy scikit-optimize jupyter
# pino_01.ipynb additionally needs: pip install torch

export DDE_BACKEND=tensorflow
jupyter lab
```

**Run order**

1. `dataset-creation/dataset.ipynb`: regenerate the laminar / turbulent splits from `data/navier-stokes/Flow.csv` (optional; the splits are already committed).
2. `Navier-Stokes/ns_01_inverse.ipynb`: train the inverse PINN on `../data/navier-stokes/Laminar_dataset.csv`, then evaluate and plot.
3. `PINO_TUTORIAL/pino_deepxde.ipynb`: train the PINO and evaluate for different `(P_in, P_out)`.
4. `model-export/*.py`: export the trained weights to TFLite. Run these from the repository root.

---

## Dataset

`data/navier-stokes/Flow.csv` holds 10,000 labelled samples (`flow_type` = `laminar` or `turbulent`, 5,000 each). `Laminar_dataset.csv` and `Turbulent_dataset.csv` are the per-regime splits.

| Columns | Meaning |
|---|---|
| `t, x, y` | space-time coordinates |
| `u, v, p` | velocity components and pressure |
| `dudx, dudy, dvdx, dvdy, dudt, dvdt` | precomputed derivatives |

The laminar set covers $x\in[0,1]$ (20 values), $y\in[0,0.1]$ (10 values) and $t\in[0,0.1]$ (10 values). <!-- TODO: state where this dataset came from (contest-provided / your own CFD run) and how it was generated. -->

---

## Limitations & known issues

- **Pressure scale is not identified in the inverse PINN.** Pressure is withheld from training while viscosity is trainable, so the momentum equation can be satisfied by trading a smaller $\nu$ against a smaller pressure gradient. In the committed run the recovered $\partial p/\partial x \approx -0.13$, whereas a linear fit to the dataset's pressure gives about $-7.3$. This is also why the relative L2 error on $p$ is ≈ 1. Fix the viscosity, or add a few pressure observations, before relying on absolute pressure values or pressure-drop thresholds.
- **$v$ relative error is not meaningful here.** The transverse velocity is near zero everywhere, so its relative L2 error is ill-conditioned; use absolute error instead.
- **No constriction geometry yet.** Both setups model a straight channel; constriction localisation from the pressure profile is the planned next step.
- **Laminar regime only.** Turbulent samples are in the dataset but are not used for training.
- **Reproducibility.** Notebook outputs reflect specific training runs; random seeds are not fixed.

---

## Roadmap

- [ ] Add a constricted-pipe geometry and show detection of constriction location and severity from $\partial p/\partial x$
- [ ] Wall-shear-stress estimate $\tau_w = \mu\,\partial u/\partial n$ at the walls
- [ ] Fix viscosity (or observe $p$) in the inverse PINN and re-validate pressure magnitude
- [ ] Non-Newtonian (blood-like) rheology
- [ ] Report on-device latency, memory and accuracy for the TFLite models
- [ ] Explore RANS-PINN-style training (data-only warm-up, then PDE loss with per-term normalisation) for harder flows

---

## References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 2019.
2. L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis. *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.* Nature Machine Intelligence, 2021.
3. L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. *DeepXDE: A deep learning library for solving differential equations.* SIAM Review, 2021. [arXiv:1907.04502](https://arxiv.org/abs/1907.04502)
4. Z. Li et al. *Physics-Informed Neural Operator for Learning Partial Differential Equations.* 2021.
5. S. Ghosh, A. Chakraborty, G. O. Brikis, B. Dey. *RANS-PINN based Simulation Surrogates for Predicting Turbulent Flows.* [arXiv:2306.06034](https://arxiv.org/abs/2306.06034)
6. S. Ghosh, A. Chakraborty, G. O. Brikis, B. Dey. *Using Parametric PINNs for Predicting Internal and External Turbulent Flows.* [arXiv:2410.18917](https://arxiv.org/abs/2410.18917)

The notebooks in `pinns-pino-examples/` follow the style of the DeepXDE documentation examples. <!-- TODO: confirm and credit the exact sources, or delete this line if they are entirely your own. -->

---

## Author

**Srimukha Sarma**, B.Tech EEE, NIT Calicut. [GitHub](https://github.com/srimukha-sarma06)

## License

<!-- TODO: add a LICENSE file (MIT is a common choice) and name it here. -->