"""Run VPINN (Helmholtz_vpinn.py) standalone and save results for comparison."""
# ruff: noqa: E402
import os
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sympy as sp
from flax import nnx

from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.arguments import TestFunction
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.operators import Dot, Grad
from jaxfun.pinns import FlaxFunction, Loss, MLPSpace, Trainer
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import adam, lbfgs
from jaxfun.utils.common import Domain, jacn, lambdify

domain = Domain(-1, 1)
V = FunctionSpace(
    32,
    Legendre,
    bcs={"left": {"D": 0}, "right": {"D": 0}},
    name="V",
    domain=domain,
)
W = MLPSpace(64, dims=1, rank=0, name="V")
w = FlaxFunction(W, "w", rngs=nnx.Rngs(1000))
v = TestFunction(V, name="v")

x = V.system.x
ue = (1 - x**2) * sp.cos(2 * sp.pi * x)

N = 1000
mesh = Line(domain.lower, domain.upper, key=nnx.Rngs(1000)())

xj_q = mesh.get_points_inside_domain(N, "legendre")
wj_q = mesh.get_weights_inside_domain(N, "legendre")
xb = mesh.get_points_on_domain()

fv = -Dot(Grad(w), Grad(v)) + w * v - (-Dot(Grad(ue), Grad(v)) + ue * v)
loss_fn = Loss((fv, xj_q, 0, wj_q), (w, xb, 0, 1))
trainer = Trainer(loss_fn)

opt_adam = adam(w, learning_rate=1e-3)
t0 = time.time()
trainer.train(opt_adam, 5000, epoch_print=1000)
time_adam = time.time() - t0
print(f"Time Adam {time_adam:.1f}s")

opt_lbfgs = lbfgs(w, memory_size=50, max_linesearch_steps=5)
t0 = time.time()
trainer.train(opt_lbfgs, 5000, epoch_print=1000, update_global_weights=1000)
time_lbfgs = time.time() - t0
print(f"Time LBFGS {time_lbfgs:.1f}s")

# Evaluate on fine grid
t0_plot = jnp.linspace(-1, 1, 1000)[:, None]
w_pred = w.module(t0_plot).reshape(-1)
uej = lambdify(x, ue)(t0_plot).reshape(-1)
error_vpinn = float(jnp.linalg.norm(w_pred - uej))
print(f"VPINN L2 Error = {error_vpinn:.6e}")

# Save results
np.savez(
    "helmholtz_vpinn_results.npz",
    xj=np.array(t0_plot),
    w_pred=np.array(w_pred),
    uej=np.array(uej),
    error=error_vpinn,
    time_adam=time_adam,
    time_lbfgs=time_lbfgs,
)
print("Saved: helmholtz_vpinn_results.npz")
