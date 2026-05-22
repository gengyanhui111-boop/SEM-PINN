"""
Method: Spectral LSTSQ-PINN (M3)
---------------------------------
Trial function : MLP neural network (FlaxFunction on MLPSpace)
Loss           : LSTSQ residual minimization (strong-form collocation)
                 L(theta) = (1/N) sum_i R(w; x_i)^2 + BC penalty
                 where R(w) = -w'' + w - f

Reference manufactured solution:
  ue = (1 - x^2) * exp(cos(2*pi*x)), zero Dirichlet BC at x = +-1

Usage:
  /Users/gyh/.workbuddy/binaries/python/envs/jaxfun313/bin/python3 helmholtz1D_lstsq_pinn.py
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
# Setup
# ============================================================
domain = Domain(-1, 1)
W = MLPSpace([64, 64], dims=1, rank=0, name="W")
w = FlaxFunction(W, "w", rngs=nnx.Rngs(42))

# Coordinate must be obtained from the space
x = W.system.x
ue = (1 - x**2) * sp.exp(sp.cos(2 * sp.pi * x))

# Strong-form residual: R(w) = -w'' + w - f  (Helmholtz: -u'' + u = f)
alpha = 1
f = Div(Grad(ue)) + alpha * ue
residual = Div(Grad(w)) + alpha * w - f

# Collocation and boundary points
mesh = Line(domain.lower, domain.upper, key=jax.random.PRNGKey(2024))
xj = mesh.get_points_inside_domain(1200, "legendre")   # Gauss-Lobatto
xb = mesh.get_points_on_domain()                        # boundary points

# Loss: LSTSQ collocation + BC penalty
loss_fn = Loss((residual, xj), (w, xb))
trainer = Trainer(loss_fn)

# ============================================================
# Training: Adam then L-BFGS
# ============================================================
print("=== Spectral LSTSQ-PINN (MLP trial, strong-form collocation) ===")

opt_adam = adam(w, learning_rate=1e-3)
t0 = time.time()
trainer.train(opt_adam, 5000, epoch_print=1000)
time_adam = time.time() - t0
print(f"Adam done: {time_adam:.1f}s")

opt_lbfgs = lbfgs(w, memory_size=50, max_linesearch_steps=5)
t0 = time.time()
trainer.train(opt_lbfgs, 5000, epoch_print=1000, update_global_weights=1000)
time_lbfgs = time.time() - t0
print(f"L-BFGS done: {time_lbfgs:.1f}s")

# ============================================================
# Evaluate
# ============================================================
x_eval = jnp.linspace(-1, 1, 1000)[:, None]
w_pred = w.module(x_eval).reshape(-1)
uej = lambdify(x, ue)(x_eval).reshape(-1)

l2_error  = float(jnp.linalg.norm(w_pred - uej) / jnp.sqrt(len(x_eval)))
linf_error = float(jnp.max(jnp.abs(w_pred - uej)))
total_time = time_adam + time_lbfgs

print(f"\nL2 error  = {l2_error:.6e}")
print(f"Linf error = {linf_error:.6e}")
print(f"Total time = {total_time:.1f}s")

# ============================================================
# Save
# ============================================================
np.savez(
    "helmholtz1D_lstsq_pinn_results.npz",
    xj=np.array(x_eval.reshape(-1)),
    uj=np.array(w_pred),
    uej=np.array(uej),
    l2_error=l2_error,
    linf_error=linf_error,
    time_adam=time_adam,
    time_lbfgs=time_lbfgs,
    total_time=total_time,
    method="LSTSQ-PINN",
)
print("Saved: helmholtz1D_lstsq_pinn_results.npz")
