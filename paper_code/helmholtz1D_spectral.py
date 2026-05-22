"""Step 1: Run Pure Spectral Galerkin and save results."""
import os, sys, time, pickle
import numpy as np
import jax
import jax.numpy as jnp
import sympy as sp
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.functionspace import FunctionSpace as SpecFunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre as space
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n

jax.config.update("jax_enable_x64", True)

N_spec = 80
# Manufactured solution: ue = (1-x^2)*exp(cos(2*pi*x)), zero BC at ±1
# 与 Helmholtz_vpinn.py 保持一致
x_sym = sp.Symbol("x", real=True)
ue_sym = (1 - x_sym**2) * sp.exp(sp.cos(2 * sp.pi * x_sym))

# 零 Dirichlet BC（制造解在 ±1 处均为 0）
bcs = {"left": {"D": 0.0}, "right": {"D": 0.0}}
D = SpecFunctionSpace(N_spec, space, bcs=bcs, name="D", fun_str="psi", scaling=n + 1)
v_spec = TestFunction(D, name="v")
u_spec = TrialFunction(D, name="u")
x_coord = D.system.x

ue_expr = D.system.expr_psi_to_base_scalar(ue_sym)

t0 = time.time()
A, L = inner(
    v_spec * (Div(Grad(u_spec)) + u_spec) - v_spec * (Div(Grad(ue_expr)) + ue_expr),
    sparse=True, sparse_tol=1000, return_all_items=False,
)
xj = D.mesh(kind="uniform", N=1000)
uh = jnp.linalg.solve(A.todense(), L)
uj = D.evaluate(xj, uh)
uej = lambdify(x_coord, ue_expr)(xj)
error_spec = jnp.linalg.norm(uj - uej)
time_spec = time.time() - t0

print(f"Spectral Galerkin: L2 Error = {error_spec:.6e}, Time = {time_spec:.1f}s")

# Save results
results = {
    "xj": np.array(xj),
    "uj": np.array(uj),
    "uej": np.array(uej),
    "error": float(error_spec),
    "time": time_spec,
    "method": "Spectral Galerkin",
}
import numpy as np
np.savez("helmholtz_spectral_results.npz", **{k: np.array(v) if hasattr(v, '__iter__') else v for k, v in results.items()})
print("Saved: helmholtz_spectral_results.npz")
