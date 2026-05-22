# ruff: noqa: E402
"""Spectral-VPINN for 2D Driven Cavity Flow (Lid-Driven Square Cavity).

Method: Spectral Variational Physics-Informed Neural Network
- 1D Legendre spectral test function spaces with Dirichlet BCs
- Kronecker product of 1D Vandermonde matrices → 2D test function basis
- MLP trial functions (Comp of two FlaxFunctions) for velocity (u,v) and pressure
- VPINN loss: project strong-form residuals onto test function space
- BC: strong-form penalty + pressure pinning
- Training: jaxfun's adam/lbfgs via Trainer (relies on jaxfun's Loss infrastructure)
  with a custom VPINN loss function via manual nnx.grad

Jacobian approach:
  To avoid AD-over-AD inside a single JIT (which causes XLA compilation blow-up),
  we separate concerns:
    1. Precompute or dynamically compute J(x; θ) = d_x [f(x; θ)] using jacn (jacfwd+vmap)
    2. Compute VPINN projection loss from residuals
    3. Differentiate total loss w.r.t. θ using nnx.value_and_grad

  This mirrors how jaxfun's own Residual framework works internally.

Outputs:
- Streamline plot, loss convergence, u/v/p contour plots
- Cached data (.npz)
"""

import os
import sys
import time

os.environ["MPLBACKEND"] = "Agg"

PLOT_ONLY = "--plot-only" in sys.argv

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize as scipy_minimize

matplotlib.use("Agg")

OUT_DIR = "/Users/gyh/WorkBuddy/20260415163831/"
DATA_FILE = OUT_DIR + "spectral_vpinn_driven_cavity_data.npz"
LOSS_FILE = OUT_DIR + "spectral_vpinn_driven_cavity_losses.npz"

# ─── Physical parameters ───────────────────────────────────────────────
Re = 100.0
nu = 2.0 / Re

# ─── Spectral test function parameters ────────────────────────────────
N_legendre = 20  # Legendre polynomial order per direction (v2: K=324)

# ─── Sampling parameters ──────────────────────────────────────────────
N_sample = 30  # quadrature points per direction (v2)

# ─── Network architecture (trial functions) ────────────────────────────
HIDDEN_V = [32, 32, 32, 32]
HIDDEN_P = [32, 32, 32, 32]

# ─── Training parameters ──────────────────────────────────────────────
ADAM_EPOCHS = 2000
LBFGS_EPOCHS = 20000  # v2: increased for deeper convergence
SEED = 2002

# ─── Loss weights ─────────────────────────────────────────────────────
BC_WEIGHT = 10.0
P_WEIGHT = 10.0


def jacn(fun, k=1):
    """Vectorized k-th order Jacobian using jacfwd/jacrev alternation + vmap."""
    for i in range(k):
        fun = jax.jacfwd(fun) if (i % 2 == 0) else jax.jacrev(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0) if k > 0 else fun


if not PLOT_ONLY:
    from flax import nnx

    from jaxfun.galerkin import FunctionSpace
    from jaxfun.galerkin.Legendre import Legendre
    from jaxfun.pinns.mesh import Rectangle
    from jaxfun.pinns.module import Comp, FlaxFunction
    from jaxfun.pinns.nnspaces import MLPSpace

    print("JAX running on", jax.devices()[0].platform.upper())
    print(f"Re = {Re}, nu = {nu}")
    print(f"Legendre test functions: N = {N_legendre}")
    print(f"Quadrature: {N_sample} x {N_sample}")
    print(f"Velocity MLP: {HIDDEN_V}, Pressure MLP: {HIDDEN_P}")

    # ─── 1D Legendre spectral spaces for test functions ─────────────
    Lx = FunctionSpace(
        N_legendre, Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        domain=(-1, 1), name="Lx",
    )
    Ly = FunctionSpace(
        N_legendre, Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        domain=(-1, 1), name="Ly",
    )

    # ─── Build 2D test function Vandermonde via Kronecker product ──
    mesh = Rectangle(-1, 1, -1, 1)

    xj_1d, wj_x = Lx.quad_points_and_weights(N_sample)
    yj_1d, wj_y = Ly.quad_points_and_weights(N_sample)

    # 2D quadrature grid and outer-product weights
    xx_2d, yy_2d = jnp.meshgrid(xj_1d, yj_1d, indexing="ij")
    xq = jnp.column_stack((xx_2d.ravel(), yy_2d.ravel()))
    wq = jnp.outer(wj_x, wj_y).ravel()
    Nq = xq.shape[0]

    Vx0 = Lx.evaluate_basis_derivative(xj_1d, k=0)   # (Nq_1d, dim_x)
    Vy0 = Ly.evaluate_basis_derivative(yj_1d, k=0)   # (Nq_1d, dim_y)
    Phi_2d = jnp.kron(Vx0, Vy0)  # (Nq, K) — 2D test function Vandermonde

    K_test = Phi_2d.shape[1]
    print(f"2D test function space: K = {K_test} ({Lx.dim} x {Ly.dim})")
    print(f"Quadrature points: {Nq}")

    # ─── Neural network trial function spaces ──────────────────────
    V_mlp = MLPSpace(HIDDEN_V, dims=2, rank=1, name="V")
    Q_mlp = MLPSpace(HIDDEN_P, dims=2, rank=0, name="Q")

    u = FlaxFunction(V_mlp, "u", rngs=nnx.Rngs(SEED))
    p = FlaxFunction(Q_mlp, "p", rngs=nnx.Rngs(SEED))

    module = Comp(u, p)

    # ─── Boundary points ──────────────────────────────────────────
    xyb = mesh.get_points_on_domain(40, 40, "random")
    Nb = xyb.shape[0]
    xyp = jnp.array([[0.0, 0.0]])
    print(f"Boundary collocation points: {Nb}")

    # ─── Precompute BC values: [u_bc, v_bc] at each boundary point ─
    # Top wall (y≈1): u = (1-x^2)^2, v = 0
    # All others: u = 0, v = 0
    u_bc = np.where(xyb[:, 1] >= 1.0 - 1e-12, (1.0 - xyb[:, 0] ** 2) ** 2, 0.0)
    v_bc = np.zeros(Nb)
    bc_vals = jnp.array(np.column_stack([u_bc, v_bc]))  # (Nb, 2)

    print(f"BC top wall points: {(xyb[:, 1] >= 1.0 - 1e-12).sum()}")

    # ─── Build graphdef/state for stateless forward pass ───────────
    # Extract submodule graphdefs for splitting
    gd_u, st_u = nnx.split(u.module, nnx.Param)
    gd_p, st_p = nnx.split(p.module, nnx.Param)
    flat_u0, unf_u = jax.flatten_util.ravel_pytree(st_u)
    flat_p0, unf_p = jax.flatten_util.ravel_pytree(st_p)
    split_idx = flat_u0.shape[0]

    # Concatenated initial flat params
    x0_flat = jnp.concatenate([flat_u0, flat_p0])
    n_params = x0_flat.shape[0]
    print(f"Total trainable parameters: {n_params}")

    def _fwd(x_, flat_params):
        """Stateless forward: returns (N, 3) = [u, v, p] given flat params."""
        fu = flat_params[:split_idx]
        fp = flat_params[split_idx:]
        mod_u = nnx.merge(gd_u, unf_u(fu))
        mod_p = nnx.merge(gd_p, unf_p(fp))
        return jnp.hstack([mod_u(x_), mod_p(x_)])

    # ─── NS residuals via jacn (separate x-AD from θ-AD) ──────────
    def compute_ns_residuals(flat_params, xpts):
        """Compute NS strong-form residuals at xpts using jacn (jit-able).

        This function uses jacfwd+vmap on x (forward AD over input),
        which is much cheaper than nested AD over both x and θ together.

        Args:
            flat_params: flat θ vector
            xpts: (N, 2) spatial points

        Returns:
            Rx, Ry, R_div: each (N,) residual at xpts
        """
        # Bind params to get a function only of x
        fwd_x = lambda x_: _fwd(x_, flat_params)

        vals = fwd_x(xpts)           # (N, 3)
        u_v = vals[:, 0]
        v_v = vals[:, 1]

        j1 = jacn(fwd_x, 1)(xpts)   # (N, 3, 2)
        ux, uy = j1[:, 0, 0], j1[:, 0, 1]
        vx, vy = j1[:, 1, 0], j1[:, 1, 1]
        px, py = j1[:, 2, 0], j1[:, 2, 1]

        j2 = jacn(fwd_x, 2)(xpts)   # (N, 3, 2, 2)
        uxx = j2[:, 0, 0, 0]
        uyy = j2[:, 0, 1, 1]
        vxx = j2[:, 1, 0, 0]
        vyy = j2[:, 1, 1, 1]

        Rx = u_v * ux + v_v * uy - nu * (uxx + uyy) + px
        Ry = u_v * vx + v_v * vy - nu * (vxx + vyy) + py
        R_div = ux + vy

        return Rx, Ry, R_div

    def total_loss_fn(flat_params):
        """Full VPINN loss as a function of flat θ."""
        # 1. NS residuals at quadrature points
        Rx, Ry, R_div = compute_ns_residuals(flat_params, xq)

        # 2. VPINN projection
        phi_Rx = Phi_2d.T @ (wq * Rx)    # (K,)
        phi_Ry = Phi_2d.T @ (wq * Ry)    # (K,)
        phi_Rd = Phi_2d.T @ (wq * R_div) # (K,)

        loss_vpinn = (phi_Rx ** 2 + phi_Ry ** 2 + phi_Rd ** 2).mean()

        # 3. BC penalty
        vals_bc = _fwd(xyb, flat_params)   # (Nb, 3)
        vel_bc = vals_bc[:, :2]            # (Nb, 2)
        loss_bc = BC_WEIGHT * ((vel_bc - bc_vals) ** 2).sum() / Nb

        # 4. Pressure pinning
        p_orig = _fwd(xyp, flat_params)[0, 2]
        loss_p = P_WEIGHT * p_orig ** 2

        return loss_vpinn + loss_bc + loss_p

    # ─── Grad function using jax.value_and_grad ────────────────────
    val_and_grad = jax.value_and_grad(total_loss_fn)

    # ─── Adam training ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Adam optimizer ({ADAM_EPOCHS} epochs)")
    print(f"{'='*60}")

    lr = 1e-3
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    x_flat = x0_flat.copy()
    m_adam = jnp.zeros_like(x_flat)
    v_adam = jnp.zeros_like(x_flat)
    adam_losses = []

    t0 = time.time()
    for epoch in range(1, ADAM_EPOCHS + 1):
        loss_val, grads = val_and_grad(x_flat)

        m_adam = beta1 * m_adam + (1 - beta1) * grads
        v_adam = beta2 * v_adam + (1 - beta2) * grads ** 2
        m_hat = m_adam / (1 - beta1 ** epoch)
        v_hat = v_adam / (1 - beta2 ** epoch)

        x_flat = x_flat - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        loss_float = float(loss_val)
        adam_losses.append(loss_float)

        if epoch % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Adam epoch {epoch}/{ADAM_EPOCHS}, loss: {loss_float:.6e}, "
                  f"elapsed: {elapsed:.0f}s")

    adam_time = time.time() - t0
    print(f"Adam finished in {adam_time:.1f}s, final loss: {adam_losses[-1]:.6e}")

    # ─── L-BFGS training ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: L-BFGS optimizer ({LBFGS_EPOCHS} iterations)")
    print(f"{'='*60}")

    lbfgs_losses = []
    call_count = [0]

    def lbfgs_callback(xk):
        call_count[0] += 1
        loss_val = float(total_loss_fn(jnp.array(xk)))
        lbfgs_losses.append(loss_val)
        if call_count[0] % 1000 == 0:
            print(f"  L-BFGS step {call_count[0]}, loss: {loss_val:.6e}")
        return False

    def loss_for_scipy(x_np):
        x_jax = jnp.array(x_np)
        loss, g = val_and_grad(x_jax)
        return float(loss), np.array(g)

    t1 = time.time()
    result = scipy_minimize(
        loss_for_scipy,
        np.array(x_flat),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": LBFGS_EPOCHS,
            "maxfun": LBFGS_EPOCHS * 20,
            "ftol": 0,
            "gtol": 1e-14,
            "disp": False,
        },
        callback=lbfgs_callback,
    )

    # Update module with final parameters
    final_flat = jnp.array(result.x)
    final_loss = float(total_loss_fn(final_flat))
    if len(lbfgs_losses) == 0:
        lbfgs_losses = [final_loss]

    # Sync module with final params for post-processing
    final_state_u = unf_u(final_flat[:split_idx])
    final_state_p = unf_p(final_flat[split_idx:])
    nnx.update(u.module, final_state_u)
    nnx.update(p.module, final_state_p)

    lbfgs_time = time.time() - t1
    total_time = adam_time + lbfgs_time
    print(f"L-BFGS finished: {result.message} (nit={result.nit}, nfev={result.nfev})")
    print(f"L-BFGS time: {lbfgs_time:.1f}s, final loss: {final_loss:.6e}")
    print(f"\nTotal training time: {total_time:.1f}s")

    # ─── Post-processing ──────────────────────────────────────────
    print("\nPost-processing...")
    N_eval = 100
    yj = jnp.linspace(-1, 1, N_eval)
    xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
    z = jnp.column_stack((xx.ravel(), yy.ravel()))

    # Use final_flat for evaluation
    uvp = _fwd(z, final_flat)

    U = np.array(uvp[:, 0].reshape(xx.shape))
    V_vel = np.array(uvp[:, 1].reshape(xx.shape))
    P = np.array(uvp[:, 2].reshape(xx.shape))
    xx_np = np.array(xx)
    yy_np = np.array(yy)

    # ─── Save data ────────────────────────────────────────────────
    np.savez(
        DATA_FILE,
        xx=xx_np, yy=yy_np, U=U, V=V_vel, P=P,
        Re=Re, nu=nu,
        N_legendre=N_legendre, N_sample=N_sample,
        hidden_v=str(HIDDEN_V), hidden_p=str(HIDDEN_P),
        adam_epochs=ADAM_EPOCHS, lbfgs_epochs=LBFGS_EPOCHS,
        adam_time=adam_time, lbfgs_time=lbfgs_time,
        total_time=total_time,
        final_loss=final_loss,
    )
    print(f"Data saved to {DATA_FILE}")

    np.savez(
        LOSS_FILE,
        adam_losses=np.array(adam_losses),
        lbfgs_losses=np.array(lbfgs_losses),
    )
    print(f"Loss history saved to {LOSS_FILE}")

    # Store for summary
    Nq_save = Nq
    K_test_save = K_test

else:
    print(f"Loading cached data from {DATA_FILE}")
    data = np.load(DATA_FILE)
    xx_np = data["xx"]
    yy_np = data["yy"]
    U = data["U"]
    V_vel = data["V"]
    P = data["P"]
    Re = float(data["Re"])
    N_legendre = int(data["N_legendre"])
    N_eval = xx_np.shape[0]

    print(f"Loading loss history from {LOSS_FILE}")
    loss_data = np.load(LOSS_FILE)
    adam_losses = list(loss_data["adam_losses"])
    lbfgs_losses = list(loss_data["lbfgs_losses"])

    Nq_save = N_sample ** 2
    K_test_save = (N_legendre - 2) ** 2  # rough estimate for composite space

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

speed = np.sqrt(U ** 2 + V_vel ** 2)

# ─── Figure 1: Loss convergence history ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Spectral-VPINN — Driven Cavity Flow (Re={Re:.0f}, N={N_legendre})",
    fontsize=14, fontweight="bold",
)

ax = axes[0]
adam_epochs_arr = np.arange(1, len(adam_losses) + 1)
lbfgs_epochs_arr = np.arange(len(adam_losses) + 1, len(adam_losses) + len(lbfgs_losses) + 1)

ax.semilogy(adam_epochs_arr, adam_losses, "b-", linewidth=0.8, alpha=0.8, label="Adam")
ax.semilogy(lbfgs_epochs_arr, lbfgs_losses, "r-", linewidth=0.8, alpha=0.8, label="L-BFGS")
ax.axvline(x=len(adam_losses) + 0.5, color="gray", linestyle="--", alpha=0.5, label="Phase switch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Loss Convergence History")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

ax = axes[1]
adam_tail = max(0, len(adam_losses) - 1000)
adam_x = np.arange(adam_tail + 1, len(adam_losses) + 1)
ax.semilogy(adam_x, adam_losses[adam_tail:], "b-", linewidth=0.8, alpha=0.8, label="Adam (tail)")
ax.semilogy(lbfgs_epochs_arr, lbfgs_losses, "r-", linewidth=0.8, alpha=0.8, label="L-BFGS")
ax.axvline(x=len(adam_losses) + 0.5, color="gray", linestyle="--", alpha=0.5, label="Phase switch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")
ax.set_title("Loss Convergence (Zoomed)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
fig.savefig(OUT_DIR + "spectral_vpinn_loss_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Loss convergence plot saved.")

# ─── Figure 2: Streamlines + Speed contour ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Spectral-VPINN — Driven Cavity (Re={Re:.0f}, N={N_legendre})",
    fontsize=14, fontweight="bold",
)

strm = axes[0].streamplot(
    xx_np[:, 0], yy_np[0, :],
    U.T, V_vel.T,
    color=speed.T, cmap="jet",
    density=1.5, linewidth=1.2, arrowsize=1.0,
    arrowstyle="->", minlength=0.15,
)
axes[0].set_title("Streamlines")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect("equal")
plt.colorbar(strm.lines, ax=axes[0], label="|U|", shrink=0.8)

cf = axes[1].contourf(xx_np, yy_np, speed, 60, cmap="jet")
axes[1].set_title("Speed |U|")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect("equal")
plt.colorbar(cf, ax=axes[1], shrink=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR + "spectral_vpinn_streamlines.png", dpi=150, bbox_inches="tight")
plt.close()
print("Streamline plot saved.")

# ─── Figure 3: u, v, p contour plots ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Spectral-VPINN — Driven Cavity (Re={Re:.0f}, N={N_legendre}) Field Variables",
    fontsize=14, fontweight="bold",
)

for i, (ax, label) in enumerate(zip(axes, ["u (x-velocity)", "v (y-velocity)", "p (pressure)"])):
    field = np.array([U, V_vel, P][i])
    levels = 100
    im = ax.contourf(xx_np, yy_np, field, levels, cmap="RdBu_r" if i < 2 else "coolwarm")
    ax.set_title(label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR + "spectral_vpinn_field_contours.png", dpi=150, bbox_inches="tight")
plt.close()
print("Field contour plot saved.")

# ─── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Spectral-VPINN Summary")
print(f"{'='*60}")
print(f"  Re = {Re}")
print(f"  Test functions: Legendre BC, N = {N_legendre}")
print(f"  Trial functions: Velocity MLP {HIDDEN_V}, Pressure MLP {HIDDEN_P}")
print(f"  Quadrature: {N_sample} x {N_sample} ({Nq_save} points)")
print(f"  Test space dim K = {K_test_save}")
print(f"  Adam: {len(adam_losses)} epochs")
print(f"  L-BFGS: {len(lbfgs_losses)} epochs")
if not PLOT_ONLY:
    print(f"  Adam time: {adam_time:.1f}s")
    print(f"  L-BFGS time: {lbfgs_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
print(f"  Final loss: {lbfgs_losses[-1]:.6e}")
print(f"  Max |u|: {np.abs(U).max():.6f}")
print(f"  Max |v|: {np.abs(V_vel).max():.6f}")
print(f"  Max |p|: {np.abs(P).max():.6f}")
print(f"{'='*60}")

print("\nSUCCESS")
