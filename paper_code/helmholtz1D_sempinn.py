"""
SEM-PINN approach using multiple sequential VPINN losses in jaxfun.

Strategy: Run VPINN on each sub-element sequentially, accumulating
the loss and gradients. This avoids the coordinate symbol conflict
while leveraging jaxfun's Loss infrastructure for correct VPINN computation.

PDE: -u'' + u = f  on [-1, 1],  u(-1) = u(1) = 0
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
from jaxfun.pinns.optimizer import adam, lbfgs
from jaxfun.utils.common import Domain, lambdify

# ============================================================
# Parameters
# ============================================================
K_ELEM = 4
N_TEST = 20
N_INT = 64
MLP_WIDTH = 64
DOMAIN = Domain(-1, 1)
elem_bounds = np.linspace(-1, 1, K_ELEM + 1)

print(f"SEM-PINN: K={K_ELEM}, N={N_TEST}, N_int={N_INT}, MLP={MLP_WIDTH}")

# ============================================================
# Global test space (must be created first for coordinate symbol)
# ============================================================
V_global = FunctionSpace(
    N_TEST * 2, Legendre.Legendre,
    bcs={"left": {"D": 0}, "right": {"D": 0}},
    name="V", domain=DOMAIN,
)
print(f"Global test space: dim = {V_global.dim}")

# Manufactured solution (using V_global.system.x for coordinate consistency)
x_sym = V_global.system.x
ue_sym = (1 - x_sym**2) * sp.exp(sp.cos(2 * sp.pi * x_sym))
ue_jax = lambdify(x_sym, ue_sym)

# ============================================================
# Global MLP
# ============================================================
W = MLPSpace(MLP_WIDTH, dims=1, rank=0, name="W")
w = FlaxFunction(W, "w", rngs=nnx.Rngs(1000))
v_global = TestFunction(V_global, name="v")

# Mesh for integration (using piecewise Legendre-Gauss quadrature)
mesh_full = Line(DOMAIN.lower, DOMAIN.upper, key=nnx.Rngs(1000)())
xb = mesh_full.get_points_on_domain()

# Build integration points: N_INT per element, piecewise
all_xj = []
all_wj = []
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

# Weak form (same structure as jaxfun VPINN example):
# R(w) = -w'' + w - f = 0  =>  -w'' + w = f = -ue'' + ue
# So: (-w'' + w - (-ue'' + ue)) * v = 0
# After IBP: (-w'*v' + w*v + ue'*v' - ue*v) = 0
# jaxfun: fv = -Dot(Grad(w), Grad(v)) + w*v - (-Dot(Grad(ue), Grad(v)) + ue*v)
fv = -Dot(Grad(w), Grad(v_global)) + w * v_global - (-Dot(Grad(ue_sym), Grad(v_global)) + ue_sym * v_global)

# Build Loss
loss_fn = Loss((fv, xj, 0, wj), (w, xb, 0, 1))
trainer = Trainer(loss_fn)

# ============================================================
# Training
# ============================================================
print("\n=== Training ===")

# Adam
print("\n--- Adam ---")
opt_adam = adam(w, learning_rate=1e-3)
t0 = time.time()
trainer.train(opt_adam, 5000, epoch_print=1000)
time_adam = time.time() - t0
print(f"  Time: {time_adam:.1f}s")

# L-BFGS
print("\n--- L-BFGS ---")
opt_lbfgs = lbfgs(w, memory_size=50, max_linesearch_steps=5)
t0 = time.time()
trainer.train(opt_lbfgs, 5000, epoch_print=1000, update_global_weights=1000)
time_lbfgs = time.time() - t0
print(f"  Time: {time_lbfgs:.1f}s")

# ============================================================
# Evaluate
# ============================================================
print("\n--- Results ---")
x_eval = jnp.linspace(-1, 1, 1000)[:, None]
w_pred = w.module(x_eval).reshape(-1)
ue_eval = ue_jax(x_eval.reshape(-1))

l2 = float(jnp.linalg.norm((w_pred - ue_eval) / len(x_eval)))
linf = float(jnp.max(jnp.abs(w_pred - ue_eval)))
rms = float(jnp.sqrt(jnp.mean((w_pred - ue_eval) ** 2)))

print(f"L2:    {l2:.6e}")
print(f"Linf:  {linf:.6e}")
print(f"RMS:   {rms:.6e}")
print(f"Total: {time_adam + time_lbfgs:.1f}s")

# Save
np.savez("helmholtz_sempinn_results.npz",
         xj=np.array(x_eval), w_pred=np.array(w_pred), uej=np.array(ue_eval),
         l2_error=l2, linf_error=linf, elem_bounds=np.array(elem_bounds))
print("\nSaved: helmholtz_sempinn_results.npz")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
ax = axes[0]
ax.plot(x_eval, ue_eval, "k-", lw=2, label="Exact")
ax.plot(x_eval, w_pred, "r--", lw=1.5, label="SEM-PINN (jaxfun)")
ax.set_xlabel("x"); ax.set_ylabel("u(x)")
ax.set_title(f"SEM-PINN: K={K_ELEM}, N_global={N_TEST*2}, N_int/elem={N_INT}\n"
             f"L2={l2:.2e}, Linf={linf:.2e}")
ax.legend(); ax.grid(True, alpha=0.3)
for b in elem_bounds:
    ax.axvline(b, color="blue", ls=":", alpha=0.5)

ax = axes[1]
ax.semilogy(x_eval, np.abs(w_pred - ue_eval) + 1e-20, "b-")
ax.set_xlabel("x"); ax.set_ylabel("|error|")
ax.set_title("Pointwise error")
ax.grid(True, alpha=0.3)
for b in elem_bounds:
    ax.axvline(b, color="blue", ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("helmholtz_sempinn_result.png", dpi=150)
print("Saved: helmholtz_sempinn_result.png")
