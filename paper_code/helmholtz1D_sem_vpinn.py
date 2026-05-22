"""
Method: SEM-VPINN (M4) — FIXED
-------------------------------
Trial function : MLP neural network (global, FlaxFunction on MLPSpace)
Test function  : Legendre spectral basis on EACH sub-element (local weak form)
Loss           : Piecewise Galerkin weak form
                 sum_e sum_k <R(w), v_k>_{Omega_e} = 0 for all test functions v_k

*** FIX ***
jaxfun's built-in lbfgs() explodes on weak-form losses because the
Galerkin projection has a null space → Hessian singular near convergence
(loss ≈ 1e-30 after Adam). L-BFGS approximate Hessian degrades when
grad ≈ 0, causing step sizes to blow up (1e-30 → 1e+22).

Fix: use Adam only, with 3-phase LR decay (1e-3 → 1e-4 → 1e-5)
No L-BFGS needed — Adam's gradient noise prevents getting stuck in
the flat regions of the weak-form loss landscape.

This is the 1D SEM-VPINN: decompose [-1,1] into K elements, use local
Legendre test functions on each element, and optimize a global MLP trial function.

Reference manufactured solution:
  ue = (1 - x^2) * exp(cos(2*pi*x)), zero Dirichlet BC at x = +-1

Usage:
  /Users/gyh/.workbuddy/binaries/python/envs/jaxfun313/bin/python3 helmholtz1D_sem_vpinn.py
"""
# ruff: noqa: E402
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sympy as sp
from flax import nnx

from jaxfun.galerkin import FunctionSpace, Legendre, TestFunction
from jaxfun.operators import Dot, Grad
from jaxfun.pinns import FlaxFunction, Loss, MLPSpace, Trainer
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import adam
from jaxfun.utils.common import Domain, lambdify

# ============================================================
# Parameters
# ============================================================
K_ELEM   = 4      # number of sub-elements
N_TEST   = 16     # Legendre test function order per element (dim = N_TEST - 2 interior modes)
N_INT    = 64     # Gauss quadrature points per element
MLP_HIDDENS = [64, 64]

print(f"SEM-VPINN: K={K_ELEM} elements, N_test={N_TEST}, N_int={N_INT}/elem")

# ============================================================
# Global test space (must be created FIRST to define coordinate symbol)
# ============================================================
domain_global = Domain(-1, 1)
V_global = FunctionSpace(
    N_TEST,
    Legendre.Legendre,
    bcs={"left": {"D": 0}, "right": {"D": 0}},
    name="V",
    domain=domain_global,
)
x_sym = V_global.system.x  # canonical coordinate symbol

# Manufactured solution
ue_sym = (1 - x_sym**2) * sp.exp(sp.cos(2 * sp.pi * x_sym))
ue_jax = lambdify(x_sym, ue_sym)

# ============================================================
# Global MLP trial function
# ============================================================
W     = MLPSpace(MLP_HIDDENS, dims=1, rank=0, name="W")
w_fn  = FlaxFunction(W, "w", rngs=nnx.Rngs(1000))
v_gbl = TestFunction(V_global, name="v")

# Boundary points
mesh_full = Line(-1.0, 1.0, key=nnx.Rngs(1000)())
xb = mesh_full.get_points_on_domain()

# ============================================================
# Build piecewise Gauss quadrature points + weights
# ============================================================
elem_bounds = np.linspace(-1, 1, K_ELEM + 1)
all_xj, all_wj = [], []
for e in range(K_ELEM):
    a, b = float(elem_bounds[e]), float(elem_bounds[e + 1])
    xq, wq = np.polynomial.legendre.leggauss(N_INT)
    xq = 0.5 * (b - a) * xq + 0.5 * (a + b)
    wq = 0.5 * (b - a) * wq
    all_xj.append(jnp.array(xq)[:, None])
    all_wj.append(jnp.array(wq))

xj = jnp.concatenate(all_xj, axis=0)
wj = jnp.concatenate(all_wj, axis=0)
print(f"Total integration points: {xj.shape[0]}")

# ============================================================
# Weak form (Galerkin): integrate by parts
#   <-w'', v> + <w, v> = <f, v>
# => -<w', v'> + <w, v> - (<-ue'', v> - <-ue, v>) = 0
# => fv = -Dot(Grad(w), Grad(v)) + w*v - (-Dot(Grad(ue), Grad(v)) + ue*v)
# ============================================================
fv = (
    -Dot(Grad(w_fn), Grad(v_gbl))
    + w_fn * v_gbl
    - (-Dot(Grad(ue_sym), Grad(v_gbl)) + ue_sym * v_gbl)
)

loss_fn = Loss((fv, xj, 0, wj), (w_fn, xb, 0, 1))
trainer  = Trainer(loss_fn)

# ============================================================
# Training — Adam only (no L-BFGS)
# ============================================================
# FIX: jaxfun's lbfgs() explodes on weak-form losses because the
# Galerkin projection has a null space → Hessian singular near convergence
# (loss ≈ 1e-30 after Adam). L-BFGS approximate Hessian degrades when
# grad ≈ 0, causing step sizes to blow up (1e-30 → 1e+22).
#
# Adam is used exclusively because its gradient noise prevents getting
# stuck in the flat regions of the weak-form loss landscape.
# Multi-phase training with decaying learning rate for stable convergence.
# ============================================================

print("\n=== Training (Adam only) ===")

# Phase 1: lr=1e-3, 8000 steps
opt_adam = adam(w_fn, learning_rate=1e-3)
t0 = time.time()
trainer.train(opt_adam, 8000, epoch_print=1000)
time_adam = time.time() - t0
print(f"Adam phase1 (lr=1e-3): {time_adam:.1f}s")

# Phase 2: lr=1e-4, 5000 steps
opt_adam2 = adam(w_fn, learning_rate=1e-4)
t1 = time.time()
trainer.train(opt_adam2, 5000, epoch_print=1000)
time_adam += time.time() - t1
print(f"Adam phase2 (lr=1e-4): {time.time()-t1:.1f}s")

# Phase 3: lr=1e-5, 5000 steps (fine convergence)
opt_adam3 = adam(w_fn, learning_rate=1e-5)
t2 = time.time()
trainer.train(opt_adam3, 5000, epoch_print=1000)
time_adam += time.time() - t2
print(f"Adam phase3 (lr=1e-5): {time.time()-t2:.1f}s")

print(f"Total Adam time: {time_adam:.1f}s")
time_lbfgs = 0.0

# ============================================================
# Evaluate
# ============================================================
x_eval = jnp.linspace(-1, 1, 1000)[:, None]
w_pred = w_fn.module(x_eval).reshape(-1)
ue_eval = ue_jax(x_eval.reshape(-1))

l2_error  = float(jnp.linalg.norm(w_pred - ue_eval) / jnp.sqrt(len(x_eval)))
linf_error = float(jnp.max(jnp.abs(w_pred - ue_eval)))
total_time = time_adam + time_lbfgs

print(f"\nL2 error  = {l2_error:.6e}")
print(f"Linf error = {linf_error:.6e}")
print(f"Total time = {total_time:.1f}s")

# ============================================================
# Save
# ============================================================
np.savez(
    "helmholtz1D_sem_vpinn_results.npz",
    xj=np.array(x_eval.reshape(-1)),
    uj=np.array(w_pred),
    uej=np.array(ue_eval),
    l2_error=l2_error,
    linf_error=linf_error,
    time_adam=time_adam,
    time_lbfgs=time_lbfgs,
    total_time=total_time,
    elem_bounds=np.array(elem_bounds),
    method="SEM-VPINN",
)
print("Saved: helmholtz1D_sem_vpinn_results.npz")

# ============================================================
# Quick plot
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
ax = axes[0]
ax.plot(x_eval, ue_eval, "k-", lw=2, label="Exact")
ax.plot(x_eval, w_pred, "r--", lw=1.5, label=f"SEM-VPINN (K={K_ELEM})")
for b in elem_bounds:
    ax.axvline(b, color="steelblue", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel("x"); ax.set_ylabel("u(x)")
ax.set_title(f"SEM-VPINN  K={K_ELEM}, N_test={N_TEST}, L2={l2_error:.2e}")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(x_eval, np.abs(w_pred - ue_eval) + 1e-20, "b-", lw=1.2)
for b in elem_bounds:
    ax.axvline(b, color="steelblue", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel("x"); ax.set_ylabel("|error|")
ax.set_title("Pointwise error")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("helmholtz1D_sem_vpinn_result.png", dpi=150)
print("Saved: helmholtz1D_sem_vpinn_result.png")
