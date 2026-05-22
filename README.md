# SEM-PINN: Coupling Spectral Element and Physics-Informed Neural Network Methods for PDEs

This repository contains the implementation code and experimental data for the paper:

> **Coupling Spectral Element and Physics-Informed Neural Network Methods for Partial Differential Equations: A Comparative Study with 1D Helmholtz and 2D Lid-Driven Cavity Benchmarks**

Submitted to *Computer Methods in Applied Mechanics and Engineering* (CMAME).

## Overview

We systematically compare six hybrid trial/test formulations for physics-informed neural PDE solvers:

| Method | Trial Space | Test Space | Loss Form |
|--------|-------------|------------|-----------|
| M1 | MLP | Collocation (strong) | PDE residual (strong form) |
| M2 | Legendre spectral | Quadrature LSTSQ | Least-squares (LGL nodes) |
| M3 | MLP | Quadrature LSTSQ (SEM) | Element-wise least-squares |
| M4 | MLP | Legendre test (global) | Galerkin weak form |
| M5 | MLP | Legendre test (SEM) | Element-local Galerkin |
| M6 | Legendre spectral | MLP test | Galerkin (spectral trial + MLP test) |

Benchmarks:
- **1D Helmholtz equation** with smooth manufactured solution (exact answer known)
- **2D lid-driven cavity flow** at Re = 100 (steady incompressible Navier--Stokes)

All methods are implemented in the [jaxfun](https://github.com/jaxfun/jaxfun) framework using JAX.

## Repository Structure

```
paper_code/          # Python implementation of all six methods
├── cavity2D_m1_pure_pinn.py          # M1: Pure PINN (2D cavity)
├── cavity2D_m2_spectral_lstsq.py     # M2: Spectral LSTSQ (2D cavity)
├── cavity2D_m3_spectral_vpinn.py      # M3: Spectral VPINN (2D cavity)
├── cavity2D_m4_sem_lstsq.py          # M4: SEM-LSTSQ (2D cavity)
├── cavity2D_m5_sem_vpinn.py          # M5: SEM-VPINN (2D cavity)
├── run_m6_spectral_mlp_vpinn.py      # M6: Spectral+MLP VPINN (2D cavity)
├── helmholtz1D_compare.py            # 1D Helmholtz comparison
├── helmholtz1D_m6_spectral_mlp_vpinn.py  # M6: 1D Helmholtz
├── m7_m2_all.py                      # M2 reference solution validation
├── m7_m2_self_consistency.py         # M2 self-consistency tests
├── m7_m2_residual_test.py            # M2 out-of-sample residual test
├── m7_m2_grid_convergence.py         # M2 grid convergence study
├── generate_paper_figures.py         # Generate all paper figures
└── plot_helmholtz1D_comparison.py    # 1D Helmholtz plots

paper_data/          # Pre-computed results (NPZ format)
├── helmholtz1D_*.npz                 # 1D Helmholtz results
├── cavity2D_m*_data.npz              # 2D cavity velocity/pressure fields
├── cavity2D_m*_losses.npz            # 2D cavity loss histories
└── ...
```

## Requirements

- Python 3.10+
- [jaxfun](https://github.com/jaxfun/jaxfun) v0.1.0
- JAX v0.5.4
- NumPy, SciPy, Matplotlib

## Reproducing Results

### 1D Helmholtz
```bash
cd paper_code
python helmholtz1D_compare.py          # Run M1-M5 comparison
python helmholtz1D_m6_spectral_mlp_vpinn.py  # Run M6
python plot_helmholtz1D_comparison.py  # Generate figures
```

### 2D Lid-Driven Cavity
```bash
cd paper_code
python cavity2D_m1_pure_pinn.py        # M1: ~4.8 h
python cavity2D_m2_spectral_lstsq.py   # M2: ~0.5 h
python cavity2D_m3_spectral_vpinn.py   # M3: ~9.4 h
python cavity2D_m4_sem_lstsq.py        # M4: ~7.2 h
python cavity2D_m5_sem_vpinn.py        # M5: ~11.4 h
python run_m6_spectral_mlp_vpinn.py    # M6: ~0.13 h
```

Pre-computed NPZ files in `paper_data/` allow reproducing all figures without re-running the full experiments.

## License

This repository is provided for reproducibility of the published results. See the paper for citation details.

## Contact

Y.H. Geng — gengyanhui@hbwe.edu.cn
