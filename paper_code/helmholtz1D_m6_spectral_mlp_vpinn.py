"""
M6 1D Helmholtz: Spectral trial + MLP test (Galerkin VPINN)
=============================================================

Method: Dual of M3 (Spectral VPINN).
  M3: MLP trial + Legendre test (global weak form)
  M6: Spectral trial (Legendre) + MLP test (Galerkin projection)

PDE: -u'' + u = f,  on [-1,1], zero Dirichlet BC
Manufactured solution: ue = (1-x^2) * exp(cos(2*pi*x))

Trial space: Legendre spectral basis with N=20, zero Dirichlet BC
  → 18 trainable interior coefficients (via FlaxFunction)
Test space: K=9 fixed MLP networks (R¹→R, hidden=[16])
Loss: Galerkin projection — (1/K) Σ_k ⟨ψ_k, R⟩²
  where R(x) = -u''(x) + u(x) - f(x)

Key insight for mode-filtering: K=9 < N_dof=18, creating an explicit
null space where Galerkin loss can be near-zero while true error is large.

Output:
  paper_data/helmholtz1D_m6_results.npz
"""
# ruff: noqa: E402
import os, sys, time
os.environ["MPLBACKEND"] = "Agg"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sympy as sp
from scipy.optimize import minimize as scipy_minimize

from flax import nnx
import flax.linen as nn

from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.pinns import FlaxFunction
from jaxfun.pinns.mesh import Line
from jaxfun.utils.common import Domain, lambdify

# ============================================================
# Parameters
# ============================================================
N_LEGENDRE  = 20                  # Legendre order
K_MLP_TEST  = 9                   # number of MLP test functions
MLP_HIDDEN  = [16]                # single hidden layer per test MLP
N_QUAD      = 128                 # Gauss quadrature points
ADAM_EPOCHS = 5000
LBFGS_EPOCHS = 5000
SEED        = 2002

# Paths
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "paper_data")
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE  = os.path.join(DATA_DIR, "helmholtz1D_m6_results.npz")
LOSS_FILE = os.path.join(DATA_DIR, "helmholtz1D_m6_losses.npz")

print(f"M6 1D Helmholtz: Spectral trial (N={N_LEGENDRE}) + MLP test (K={K_MLP_TEST})")
print(f"  Quadrature: N={N_QUAD}, MLP hidden={MLP_HIDDEN}")

# ============================================================
# 1. Spectral trial space — Legendre basis with zero Dirichlet BC
# ============================================================
domain = Domain(-1, 1)
L = FunctionSpace(
    N_LEGENDRE,
    Legendre,
    bcs={"left": {"D": 0}, "right": {"D": 0}},
    name="L",
    domain=domain,
)
print(f"  Trial space dim = {L.dim}  (interior DOF = {L.dim})")

# Manufactured solution
x_sym = L.system.x
ue_sym = (1 - x_sym**2) * sp.exp(sp.cos(2 * sp.pi * x_sym))
ue_jax = lambdify(x_sym, ue_sym)

# Source term: f = -ue'' + ue
f_sym = -sp.diff(ue_sym, x_sym, 2) + ue_sym
f_jax = lambdify(x_sym, f_sym)

# Create FlaxFunction for spectral trial (trainable coefficients)
u_fn = FlaxFunction(L, "u", rngs=nnx.Rngs(SEED))

# ============================================================
# 2. MLP test functions — K fixed MLPs: R¹→R
# ============================================================
class TestMLP1D(nn.Module):
    hidden: list
    @nn.compact
    def __call__(self, x):
        for w in self.hidden:
            x = nn.tanh(nn.Dense(w)(x))
        return nn.Dense(1)(x)

rng = jr.PRNGKey(SEED + 1000)
keys = jr.split(rng, K_MLP_TEST)
mlps = []
for k in range(K_MLP_TEST):
    m = TestMLP1D(hidden=MLP_HIDDEN)
    p = m.init(keys[k], jnp.ones((1, 1)))
    mlps.append((p, m.apply))
print(f"  Built {K_MLP_TEST} MLP test functions (hidden {MLP_HIDDEN})")

def evaluate_all_mlp(mlps, x):
    """Evaluate all K MLPs at points x: (N,)→(N,K)."""
    x_in = x[:, None] if x.ndim == 1 else x
    out = []
    for params, apply_fn in mlps:
        out.append(apply_fn(params, x_in)[:, 0])
    return jnp.column_stack(out)

# ============================================================
# 3. Quadrature points — Gauss-Legendre on [-1,1]
# ============================================================
xq_raw, wq_raw = np.polynomial.legendre.leggauss(N_QUAD)
xq = jnp.array(xq_raw)
wq = jnp.array(wq_raw)
Nq = xq.shape[0]
print(f"  Quadrature points: {Nq}")

# Precompute weighted test matrix Ψ̃_{i,k} = w_i · ψ_k(x_i)
Psi_raw = evaluate_all_mlp(mlps, xq)     # (Nq, K)
Psi_w   = Psi_raw * wq[:, None]          # (Nq, K) weighted
print(f"  Test matrix Ψ: {Psi_raw.shape}")

# ============================================================
# 4. Flatten parameters
# ============================================================
gd, st = nnx.split(u_fn.module, nnx.Param)
f0, unf = jax.flatten_util.ravel_pytree(st)
flat0 = f0.copy()
print(f"  Trainable parameters: {flat0.shape[0]}")

# ============================================================
# 5. Loss function — Galerkin projection
# ============================================================
def merge_mod(flat):
    st_ = unf(flat)
    return nnx.merge(gd, st_)

def u_at_x(flat, x):
    """Evaluate u at points x: (N,)→(N,)."""
    m = merge_mod(flat)
    return m(x[:, None] if x.ndim == 1 else x)[:, 0]

def loss_fn(flat):
    """Galerkin loss: mean_k (sum_i w_i · ψ_k(x_i) · R(x_i))^2."""
    # PDE residual R(x) = -u''(x) + u(x) - f(x)
    # Compute u and its 2nd derivative at quadrature points
    def u_scalar(x_scalar):
        """x_scalar: scalar → scalar u."""
        m = merge_mod(flat)
        return m(jnp.array([x_scalar])[:, None])[0, 0]

    u_val = u_at_x(flat, xq)                     # (Nq,)
    u_xx  = jax.vmap(jax.grad(jax.grad(u_scalar)))(xq)  # (Nq,)
    f_val = f_jax(np.array(xq))                  # (Nq,)
    R = -u_xx + u_val - f_val                    # (Nq,)

    # Galerkin projection: proj_k = Σ_i w_i · ψ_k(x_i) · R(x_i)
    proj = Psi_w.T @ R                           # (K,)
    loss_gal = jnp.mean(proj**2)

    return loss_gal

loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

# ============================================================
# 6. Training — Adam
# ============================================================
print(f"\n{'='*60}")
print(f"M6 Phase 1: Adam ({ADAM_EPOCHS} epochs)")
print(f"{'='*60}")

lr = 1e-3
beta1, beta2, eps = 0.9, 0.999, 1e-8
x_flat = flat0.copy()
m_adam = jnp.zeros_like(x_flat)
v_adam = jnp.zeros_like(x_flat)
adam_losses = []

t0 = time.time()
for epoch in range(1, ADAM_EPOCHS + 1):
    loss_v, grads = loss_and_grad(x_flat)
    m_adam = beta1 * m_adam + (1 - beta1) * grads
    v_adam = beta2 * v_adam + (1 - beta2) * grads**2
    m_h = m_adam / (1 - beta1**epoch)
    v_h = v_adam / (1 - beta2**epoch)
    x_flat = x_flat - lr * m_h / (jnp.sqrt(v_h) + eps)
    adam_losses.append(float(loss_v))
    if epoch % 1000 == 0:
        print(f"  epoch {epoch:5d}/{ADAM_EPOCHS}  loss={float(loss_v):.6e}  "
              f"({time.time()-t0:.0f}s)")

adam_time = time.time() - t0
print(f"Adam done: {adam_time:.1f}s  final loss={adam_losses[-1]:.6e}")

# ============================================================
# 7. Training — L-BFGS
# ============================================================
print(f"\n{'='*60}")
print(f"M6 Phase 2: L-BFGS ({LBFGS_EPOCHS} iters)")
print(f"{'='*60}")

lbfgs_losses = []
call_cnt = [0]

def scipy_obj(params_np):
    call_cnt[0] += 1
    pj = jnp.array(params_np)
    loss_v, grad_v = loss_and_grad(pj)
    lbfgs_losses.append(float(loss_v))
    if call_cnt[0] % 1000 == 0:
        print(f"  iter {call_cnt[0]:5d}  loss={float(loss_v):.6e}")
    return float(loss_v), np.array(grad_v)

t1 = time.time()
result = scipy_minimize(
    scipy_obj,
    np.array(x_flat),
    method="L-BFGS-B",
    jac=True,
    options=dict(maxiter=LBFGS_EPOCHS, maxfun=LBFGS_EPOCHS * 20,
                 ftol=0, gtol=1e-14, disp=False),
)
lbfgs_time = time.time() - t1
final_flat = jnp.array(result.x)
total_time = adam_time + lbfgs_time

print(f"L-BFGS done: {result.message}  nit={result.nit}  nfev={result.nfev}")
print(f"L-BFGS time: {lbfgs_time:.1f}s")
print(f"Total time:   {total_time:.1f}s")

# ============================================================
# 8. Evaluation
# ============================================================
x_eval = jnp.linspace(-1, 1, 1000)
u_pred = u_at_x(final_flat, x_eval)
ue_eval = ue_jax(np.array(x_eval))

l2_error = float(jnp.linalg.norm(u_pred - ue_eval) / jnp.sqrt(len(x_eval)))
linf_error = float(jnp.max(jnp.abs(u_pred - ue_eval)))
final_loss = float(loss_fn(final_flat))

print(f"\n{'='*60}")
print(f"M6 1D Results")
print(f"{'='*60}")
print(f"  L2 error    = {l2_error:.6e}")
print(f"  Linf error  = {linf_error:.6e}")
print(f"  Final loss  = {final_loss:.6e}")
print(f"  Adam time   = {adam_time:.1f}s")
print(f"  L-BFGS time = {lbfgs_time:.1f}s")
print(f"  Total time  = {total_time:.1f}s")

# ============================================================
# 9. Save results
# ============================================================
np.savez(OUT_FILE,
         xj=np.array(x_eval),
         uj=np.array(u_pred),
         uej=np.array(ue_eval),
         l2_error=l2_error,
         linf_error=linf_error,
         final_loss=final_loss,
         time_adam=adam_time,
         time_lbfgs=lbfgs_time,
         total_time=total_time,
         N_legendre=N_LEGENDRE,
         K_mlp_test=K_MLP_TEST,
         method="M6: Spectral+MLP VPINN")
np.savez(LOSS_FILE,
         adam_losses=np.array(adam_losses),
         lbfgs_losses=np.array(lbfgs_losses))
print(f"Saved: {os.path.basename(OUT_FILE)}")
print(f"Saved: {os.path.basename(LOSS_FILE)}")
print(f"\n{'='*60}")
print("M6 1D Helmholtz complete!")
print(f"{'='*60}")
