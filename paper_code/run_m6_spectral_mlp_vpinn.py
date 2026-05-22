# ruff: noqa: E402
"""M6: Spectral MLP-VPINN for 2D Driven Cavity Flow on [0,1]x[0,1].

Method summary:
  Trial space:  Legendre spectral basis (same as M2), 1200 coefficients
  Test space:   Fixed MLP networks (not trained), K=324 test functions
  Loss:         Galerkin projection: (1/K) Σ_k ⟨ψ_k, R⟩²
  Training:     Manual Adam + scipy L-BFGS-B

M6 is the DUAL of M4 (old M3): M4 has MLP trial + Legendre test;
M6 has spectral trial + MLP test.

Output:
  paper_data/m6_spectral_mlp_vpinn_data.npz
  paper_data/m6_spectral_mlp_vpinn_losses.npz
  paper_figures/Fig M6. Loss convergence.png
  paper_figures/Fig M6. Streamlines.png
  paper_figures/Fig M6. Field contours.png
"""

import os, sys, time
os.environ["MPLBACKEND"] = "Agg"

PLOT_ONLY = "--plot-only" in sys.argv

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Output paths ───────────────────────────────────────────────────────
CODE_DIR  = os.path.dirname(os.path.abspath(__file__))  # paper_code/
ROOT_DIR  = os.path.dirname(CODE_DIR)                    # workspace root
DATA_DIR  = os.path.join(ROOT_DIR, "paper_data")
FIG_DIR   = os.path.join(ROOT_DIR, "paper_figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "m6_spectral_mlp_vpinn_data.npz")
LOSS_FILE = os.path.join(DATA_DIR, "m6_spectral_mlp_vpinn_losses.npz")

# ─── Physics ────────────────────────────────────────────────────────────
Re = 100.0
nu = 1.0 / Re

# ─── Spectral trial space (same as M2) ─────────────────────────────────
N_LEGENDRE = 20      # Legendre order per direction

# ─── MLP test functions ────────────────────────────────────────────────
K_MLP_TEST = 324      # 18x18 = matches M4's K
MLP_HIDDEN = [32, 32] # Two hidden layers per test MLP
N_QUAD = 30           # quadrature points per direction (Nq = 900)

# ─── Training ──────────────────────────────────────────────────────────
ADAM_EPOCHS  = 5000
LBFGS_EPOCHS = 12000  # reduced from 20000 to fit 10min bash timeout
SEED = 2002
BC_WEIGHT = 2.0
P_WEIGHT  = 10.0

# ═══════════════════════════════════════════════════════════════════════
# MLP test function builder
# ═══════════════════════════════════════════════════════════════════════

def build_mlp_test_functions(rng_key, K, hidden_sizes):
    """Build K fixed MLPs: R²→R. Returns list of (params, apply_fn)."""
    import flax.linen as nn

    class TestMLP(nn.Module):
        hidden: list
        @nn.compact
        def __call__(self, x):
            for w in self.hidden:
                x = nn.tanh(nn.Dense(w)(x))
            return nn.Dense(1)(x)

    keys = jr.split(rng_key, K)
    mlps = []
    for k in range(K):
        m = TestMLP(hidden=hidden_sizes)
        p = m.init(keys[k], jnp.ones((1, 2)))
        mlps.append((p, m.apply))
    return mlps


def evaluate_all_mlp(mlps, x):
    """Evaluate all K MLPs at points x: (N,2)→(N,K)."""
    out = []
    for params, apply_fn in mlps:
        out.append(apply_fn(params, x)[:, 0])  # (N,)
    return jnp.column_stack(out)               # (N, K)

# ═══════════════════════════════════════════════════════════════════════
# Training block
# ═══════════════════════════════════════════════════════════════════════

if not PLOT_ONLY:
    from flax import nnx
    from jaxfun.galerkin import FunctionSpace, TensorProduct
    from jaxfun.galerkin.Legendre import Legendre
    from jaxfun.galerkin.tensorproductspace import VectorTensorProductSpace
    from jaxfun.pinns.mesh import Rectangle
    from jaxfun.pinns.module import Comp, FlaxFunction

    print("JAX device:", jax.devices()[0].platform.upper())
    print(f"Re={Re:.0f}, Legendre N={N_LEGENDRE}, MLP test K={K_MLP_TEST}, "
          f"quad N={N_QUAD}x{N_QUAD}")

    # ─── 1. Spectral trial spaces ───────────────────────────────────
    L0 = FunctionSpace(N_LEGENDRE, Legendre, domain=(0, 1), name="L")
    S  = TensorProduct(L0, L0, name="S")            # pressure
    V  = VectorTensorProductSpace(S, name="V")       # velocity (u,v)
    print(f"  Velocity dim={V.dim}, Pressure dim={S.dim}, Total coeffs={V.dim+S.dim}")

    u_fn = FlaxFunction(V, "u", rngs=nnx.Rngs(SEED))
    p_fn = FlaxFunction(S, "p", rngs=nnx.Rngs(SEED))

    # ─── 2. MLP test functions ──────────────────────────────────────
    rng = jr.PRNGKey(SEED + 1000)
    mlps_all = build_mlp_test_functions(rng, K_MLP_TEST, MLP_HIDDEN)
    print(f"  Built {K_MLP_TEST} MLP test functions (hidden {MLP_HIDDEN})")

    # ─── 3. Quadrature points ───────────────────────────────────────
    mesh = Rectangle(0, 1, 0, 1)
    xq = mesh.get_points_inside_domain(N_QUAD, N_QUAD, "legendre")       # (Nq, 2)
    wq = mesh.get_weights_inside_domain(N_QUAD, N_QUAD, "legendre")      # (Nq,)
    Nq = xq.shape[0]
    print(f"  Quadrature points: {Nq}")

    # ─── 4. Precompute weighted test matrix Ψ̃_{i,k} = w_i · ψ_k(x_i) ─
    Psi_raw = evaluate_all_mlp(mlps_all, xq)         # (Nq, K)
    Psi_w   = Psi_raw * wq[:, None]                  # (Nq, K)  weighted
    print(f"  Test matrix Ψ: {Psi_raw.shape}")

    # ─── 5. Boundary points and BC values ───────────────────────────
    xyb = mesh.get_points_on_domain(40, 40, "random")
    Nb = xyb.shape[0]
    u_bc = jnp.where(xyb[:, 1] >= 1.0 - 1e-12,
                     16.0 * xyb[:, 0]**2 * (1.0 - xyb[:, 0])**2, 0.0)
    v_bc = jnp.zeros(Nb)
    bc_target = jnp.column_stack([u_bc, v_bc])       # (Nb, 2)
    xyp = jnp.array([[0.5, 0.5]])

    # ─── 6. Flatten parameters ─────────────────────────────────────
    gd_u, st_u = nnx.split(u_fn.module, nnx.Param)
    gd_p, st_p = nnx.split(p_fn.module, nnx.Param)
    f0_u, unf_u = jax.flatten_util.ravel_pytree(st_u)
    f0_p, unf_p = jax.flatten_util.ravel_pytree(st_p)
    split_idx = f0_u.shape[0]
    flat0 = jnp.concatenate([f0_u, f0_p])
    print(f"  Trainable parameters: {flat0.shape[0]}")

    # ─── 7. Forward helpers ────────────────────────────────────────
    def merge_mods(flat):
        st_u = unf_u(flat[:split_idx])
        st_p = unf_p(flat[split_idx:])
        return nnx.merge(gd_u, st_u), nnx.merge(gd_p, st_p)

    def uv_p_at(flat, x):
        """(N,2) → u:(N,2), p:(N,)"""
        mu, mp = merge_mods(flat)
        return mu(x), mp(x)

    # ─── 8. Loss function ──────────────────────────────────────────
    def loss_fn(flat):
        mu, mp = merge_mods(flat)

        def u_single(xx):
            """xx: (2,) → scalar u."""
            return mu(xx[None, :])[0, 0]

        def v_single(xx):
            """xx: (2,) → scalar v."""
            return mu(xx[None, :])[0, 1]

        def p_single(xx):
            """xx: (2,) → scalar p."""
            return mp(xx[None, :])[0, 0]

        u_val = mu(xq)   # (Nq, 2)
        p_val = mp(xq)   # (Nq,)

        # 1st derivatives: each → (Nq, 2)
        u_grad = jax.vmap(jax.grad(u_single))(xq)  # (Nq, 2): [du/dx, du/dy]
        v_grad = jax.vmap(jax.grad(v_single))(xq)  # (Nq, 2): [dv/dx, dv/dy]
        p_grad = jax.vmap(jax.grad(p_single))(xq)  # (Nq, 2): [dp/dx, dp/dy]
        ux, uy = u_grad[:, 0], u_grad[:, 1]
        vx, vy = v_grad[:, 0], v_grad[:, 1]
        px, py = p_grad[:, 0], p_grad[:, 1]

        # 2nd derivatives of u
        hess_u = jax.vmap(jax.hessian(u_single))(xq)  # (Nq, 2, 2)
        uxx, uyy = hess_u[:, 0, 0], hess_u[:, 1, 1]

        # 2nd derivatives of v
        hess_v = jax.vmap(jax.hessian(v_single))(xq)  # (Nq, 2, 2)
        vxx, vyy = hess_v[:, 0, 0], hess_v[:, 1, 1]

        # NS residuals
        Ru = u_val[:, 0]*ux + u_val[:, 1]*uy - nu*(uxx+uyy) + px
        Rv = u_val[:, 0]*vx + u_val[:, 1]*vy - nu*(vxx+vyy) + py
        Rd = ux + vy

        # Galerkin projection: proj_k = sum_i w_i·ψ_k(x_i)·R(x_i)
        proj_u = Psi_w.T @ Ru   # (K,)
        proj_v = Psi_w.T @ Rv
        proj_d = Psi_w.T @ Rd
        loss_gal = jnp.mean(proj_u**2 + proj_v**2 + proj_d**2)

        # ---- boundary ----
        uv_b = mu(xyb)                                    # (Nb,2)
        loss_bc = BC_WEIGHT * jnp.mean((uv_b - bc_target)**2)

        # ---- pressure pinning ----
        p0 = mp(xyp)[0, 0]
        loss_p = P_WEIGHT * p0**2

        return loss_gal + loss_bc + loss_p

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # ─── 9. Adam ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"M6 Phase 1: Adam ({ADAM_EPOCHS} epochs)")
    print(f"{'='*60}")

    lr   = 1e-3
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    x_flat = flat0.copy()
    m_adam = jnp.zeros_like(x_flat)
    v_adam = jnp.zeros_like(x_flat)
    adam_losses = []

    t0 = time.time()
    for epoch in range(1, ADAM_EPOCHS + 1):
        loss_val, grads = loss_and_grad(x_flat)
        m_adam = beta1 * m_adam + (1-beta1) * grads
        v_adam = beta2 * v_adam + (1-beta2) * grads**2
        m_h = m_adam / (1 - beta1**epoch)
        v_h = v_adam / (1 - beta2**epoch)
        x_flat = x_flat - lr * m_h / (jnp.sqrt(v_h) + eps)
        adam_losses.append(float(loss_val))
        if epoch % 1000 == 0:
            print(f"  epoch {epoch:5d}/{ADAM_EPOCHS}  loss={float(loss_val):.6e}  "
                  f"({time.time()-t0:.0f}s)")

    adam_time = time.time() - t0
    print(f"Adam done: {adam_time:.1f}s  final loss={adam_losses[-1]:.6e}")

    # ─── 10. L-BFGS ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"M6 Phase 2: L-BFGS ({LBFGS_EPOCHS} iters)")
    print(f"{'='*60}")

    lbfgs_losses = []
    call_cnt = [0]

    def scipy_obj(params_np):
        call_cnt[0] += 1
        pj = jnp.array(params_np)
        loss, grad = loss_and_grad(pj)
        lbfgs_losses.append(float(loss))
        if call_cnt[0] % 2000 == 0:
            print(f"  iter {call_cnt[0]:5d}  loss={float(loss):.6e}")
        return float(loss), np.array(grad)

    t1 = time.time()
    from scipy.optimize import minimize as scipy_minimize
    result = scipy_minimize(
        scipy_obj,
        np.array(x_flat),
        method="L-BFGS-B",
        jac=True,
        options=dict(maxiter=LBFGS_EPOCHS, maxfun=LBFGS_EPOCHS*20,
                     ftol=0, gtol=1e-14, disp=False),
    )
    lbfgs_time = time.time() - t1
    final_flat = jnp.array(result.x)
    total_time = adam_time + lbfgs_time

    print(f"L-BFGS done: {result.message}  nit={result.nit}  nfev={result.nfev}")
    print(f"L-BFGS time: {lbfgs_time:.1f}s")
    print(f"Total time:   {total_time:.1f}s ({total_time/3600:.2f}h)")

    # ─── 11. Post-processing ─────────────────────────────────────────
    print("\nPost-processing...")
    N_eval = 100
    yj = jnp.linspace(0, 1, N_eval)
    xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
    z = jnp.column_stack((xx.ravel(), yy.ravel()))

    mu_f, mp_f = merge_mods(final_flat)
    uv = mu_f(z)      # (10000, 2)
    pp = mp_f(z)      # (10000,)

    U     = np.array(uv[:, 0].reshape(xx.shape))
    V_vel = np.array(uv[:, 1].reshape(xx.shape))
    P     = np.array(pp.reshape(xx.shape))
    xx_np = np.array(xx)
    yy_np = np.array(yy)

    # Save
    np.savez(DATA_FILE,
             xx=xx_np, yy=yy_np, U=U, V=V_vel, P=P,
             Re=Re, nu=nu,
             N_legendre=N_LEGENDRE, K_mlp_test=K_MLP_TEST,
             N_quad=N_QUAD,
             adam_epochs=ADAM_EPOCHS, lbfgs_epochs=LBFGS_EPOCHS,
             adam_time=adam_time, lbfgs_time=lbfgs_time,
             total_time=total_time,
             final_loss=float(loss_fn(final_flat)))
    np.savez(LOSS_FILE,
             adam_losses=np.array(adam_losses),
             lbfgs_losses=np.array(lbfgs_losses))
    print(f"Data → {os.path.basename(DATA_FILE)}")
    print(f"Loss → {os.path.basename(LOSS_FILE)}")

else:
    # PLOT_ONLY
    data = np.load(DATA_FILE)
    xx_np, yy_np = data["xx"], data["yy"]
    U, V_vel, P   = data["U"], data["V"], data["P"]
    Re = float(data["Re"]); N_LEGENDRE = int(data["N_legendre"])
    loss_d = np.load(LOSS_FILE)
    adam_losses  = list(loss_d["adam_losses"])
    lbfgs_losses = list(loss_d["lbfgs_losses"])

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

speed = np.sqrt(U**2 + V_vel**2)
n_adam = len(adam_losses)
n_lbfgs = len(lbfgs_losses)

# ─── Fig 1: Loss convergence ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"M6: Spectral MLP-VPINN — Cavity [0,1]$^2$  "
             f"(Re={Re:.0f}, N={N_LEGENDRE}, K={K_MLP_TEST})",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.semilogy(np.arange(1, n_adam+1), adam_losses, "b-", lw=0.8, alpha=0.8, label="Adam")
if n_lbfgs:
    ax.semilogy(np.arange(n_adam, n_adam+n_lbfgs), lbfgs_losses, "r-", lw=0.8, alpha=0.8, label="L-BFGS")
ax.axvline(x=n_adam+.5, color="gray", ls="--", alpha=0.4, label="Phase switch")
ax.set(xlabel="Epoch", ylabel="Loss (log)", title="Full history")
ax.legend(fontsize=9); ax.grid(True, alpha=0.25, which="both")

ax = axes[1]
tail = max(0, n_adam-1000)
ax.semilogy(np.arange(tail+1, n_adam+1), adam_losses[tail:], "b-", lw=0.8, alpha=0.8, label="Adam tail")
if n_lbfgs:
    ax.semilogy(np.arange(n_adam, n_adam+n_lbfgs), lbfgs_losses, "r-", lw=0.8, alpha=0.8, label="L-BFGS")
ax.set(xlabel="Epoch", ylabel="Loss (log)", title="Zoomed")
ax.legend(fontsize=9); ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "Fig M6. Loss convergence.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: Fig M6. Loss convergence.png")

# ─── Fig 2: Streamlines ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"M6: Spectral MLP-VPINN — Cavity [0,1]$^2$  "
             f"(Re={Re:.0f})", fontsize=13, fontweight="bold")

strm = axes[0].streamplot(xx_np[:, 0], yy_np[0, :], U.T, V_vel.T,
                          color=speed.T, cmap="jet", density=1.5, linewidth=1.2,
                          arrowsize=1.0, arrowstyle="->", minlength=0.15)
axes[0].set(title="Streamlines", xlabel="x", ylabel="y", aspect="equal")
plt.colorbar(strm.lines, ax=axes[0], label="|U|", shrink=0.8)

cf = axes[1].contourf(xx_np, yy_np, speed, 60, cmap="jet")
axes[1].set(title="Speed |U|", xlabel="x", ylabel="y", aspect="equal")
plt.colorbar(cf, ax=axes[1], shrink=0.8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "Fig M6. Streamlines.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: Fig M6. Streamlines.png")

# ─── Fig 3: u, v, p contours ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"M6: Spectral MLP-VPINN — Field Variables  "
             f"(Re={Re:.0f})", fontsize=13, fontweight="bold")

for i, (ax, label) in enumerate(zip(axes,
    ["u (x-velocity)", "v (y-velocity)", "p (pressure)"])):
    field = [U, V_vel, P][i]
    im = ax.contourf(xx_np, yy_np, field, 100,
                     cmap="RdBu_r" if i < 2 else "coolwarm")
    ax.set(title=label, xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "Fig M6. Field contours.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: Fig M6. Field contours.png")

# ─── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"M6 Summary")
print(f"{'='*60}")
print(f"  Re={Re:.0f}  Trial: Legendre N={N_LEGENDRE}  Test: {K_MLP_TEST} MLPs")
print(f"  Adam: {n_adam} epochs  |  L-BFGS: {n_lbfgs} iters")
print(f"  Final loss: {lbfgs_losses[-1] if n_lbfgs else adam_losses[-1]:.6e}")
print(f"  |u|_max={np.abs(U).max():.6f}  |v|_max={np.abs(V_vel).max():.6f}")
print(f"  |p|_max={np.abs(P).max():.6f}")
print(f"{'='*60}\nM6 完成")
