"""
Method: SEM-LSTSQ-PINN (M5)
------------------------------
Trial function : MLP neural network (global, FlaxFunction on MLPSpace)
Loss           : LSTSQ strong-form residual on piecewise Gauss quadrature points
                 L(theta) = sum_e (1/N_e) sum_i R(w; x_i^e)^2 + BC penalty
                 where R(w) = -w'' + w - f  (Helmholtz, alpha=1)

This combines SEM element decomposition (piecewise Gauss-Lobatto quadrature)
with the strong-form LSTSQ loss (not weak form), and uses a global MLP.

Reference manufactured solution:
  ue = (1 - x^2) * exp(cos(2*pi*x)), zero Dirichlet BC at x = +-1

Usage:
  /Users/gyh/.workbuddy/binaries/python/envs/jaxfun313/bin/python3 helmholtz1D_sem_lstsq.py
"""
# ruff: noqa: E402
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sympy as sp
from flax import nnx

from jaxfun.pinns import FlaxFunction, Loss, MLPSpace, Trainer
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import adam, lbfgs
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import Domain, lambdify

# ============================================================
# Parameters
# ============================================================
K_ELEM      = 4     # number of sub-elements
N_INT       = 64    # Gauss quadrature points per element
MLP_HIDDENS = [64, 64]

print(f"SEM-LSTSQ: K={K_ELEM} elements, N_int={N_INT}/elem, MLP={MLP_HIDDENS}")

# ============================================================
# MLP trial function (coordinate symbol from MLPSpace)
# ============================================================
domain = Domain(-1, 1)
W     = MLPSpace(MLP_HIDDENS, dims=1, rank=0, name="W")
w_fn  = FlaxFunction(W, "w", rngs=nnx.Rngs(42))

x_sym = W.system.x   # canonical coordinate symbol
ue_sym = (1 - x_sym**2) * sp.exp(sp.cos(2 * sp.pi * x_sym))
ue_jax = lambdify(x_sym, ue_sym)

# Strong-form residual: R(w) = -w'' + w - f
alpha = 1
f = Div(Grad(ue_sym)) + alpha * ue_sym
residual = Div(Grad(w_fn)) + alpha * w_fn - f

# ============================================================
# Piecewise Gauss-Legendre collocation points
# ============================================================
elem_bounds = np.linspace(-1, 1, K_ELEM + 1)
all_xj = []
for e in range(K_ELEM):
    a, b = float(elem_bounds[e]), float(elem_bounds[e + 1])
    xq, _ = np.polynomial.legendre.leggauss(N_INT)
    xq = 0.5 * (b - a) * xq + 0.5 * (a + b)
    all_xj.append(jnp.array(xq)[:, None])

xj = jnp.concatenate(all_xj, axis=0)
print(f"Total collocation points: {xj.shape[0]}")

# Boundary points
mesh_full = Line(-1.0, 1.0, key=jax.random.PRNGKey(0))
xb = mesh_full.get_points_on_domain()

# ============================================================
# Loss: strong-form LSTSQ + BC
# ============================================================
loss_fn = Loss((residual, xj), (w_fn, xb))
trainer  = Trainer(loss_fn)

# ============================================================
# Training
# ============================================================
print("\n=== Training ===")
opt_adam = adam(w_fn, learning_rate=1e-3)
t0 = time.time()
trainer.train(opt_adam, 5000, epoch_print=1000)
time_adam = time.time() - t0
print(f"Adam: {time_adam:.1f}s")

opt_lbfgs = lbfgs(w_fn, memory_size=50, max_linesearch_steps=5)
t0 = time.time()
trainer.train(opt_lbfgs, 5000, epoch_print=1000, update_global_weights=1000)
time_lbfgs = time.time() - t0
print(f"L-BFGS: {time_lbfgs:.1f}s")

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
    "helmholtz1D_sem_lstsq_results.npz",
    xj=np.array(x_eval.reshape(-1)),
    uj=np.array(w_pred),
    uej=np.array(ue_eval),
    l2_error=l2_error,
    linf_error=linf_error,
    time_adam=time_adam,
    time_lbfgs=time_lbfgs,
    total_time=total_time,
    elem_bounds=np.array(elem_bounds),
    method="SEM-LSTSQ",
)
print("Saved: helmholtz1D_sem_lstsq_results.npz")

# ============================================================
# Quick plot
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
ax = axes[0]
ax.plot(x_eval, ue_eval, "k-", lw=2, label="Exact")
ax.plot(x_eval, w_pred, "r--", lw=1.5, label=f"SEM-LSTSQ (K={K_ELEM})")
for b in elem_bounds:
    ax.axvline(b, color="steelblue", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel("x"); ax.set_ylabel("u(x)")
ax.set_title(f"SEM-LSTSQ  K={K_ELEM}, N_int={N_INT}/elem, L2={l2_error:.2e}")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(x_eval, np.abs(w_pred - ue_eval) + 1e-20, "b-", lw=1.2)
for b in elem_bounds:
    ax.axvline(b, color="steelblue", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel("x"); ax.set_ylabel("|error|")
ax.set_title("Pointwise error")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("helmholtz1D_sem_lstsq_result.png", dpi=150)
print("Saved: helmholtz1D_sem_lstsq_result.png")
