"""
generate_paper_figures_v7.py
Generate all paper figures including M6 as a parallel method.
6 methods: M1-M6 (M6: Spectral + MLP VPINN)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

OUT = 'paper_figures'
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
data = {}
DATA_DIR = "/Users/gyh/WorkBuddy/20260415163831/paper_data"

specs = {
    'M1': (f'{DATA_DIR}/cavity2D_m1_data.npz',        'M1: Pure PINN'),
    'M2': (f'{DATA_DIR}/cavity2D_m2_data.npz',        'M2: Spectral LSTSQ'),
    'M3': (f'{DATA_DIR}/cavity2D_m4_data.npz',        'M3: SEM-LSTSQ'),
    'M4': (f'{DATA_DIR}/cavity2D_m3_data.npz',        'M4: Spectral VPINN'),
    'M5': (f'{DATA_DIR}/cavity2D_m5_data.npz',        'M5: SEM-VPINN'),
    'M6': (f'{DATA_DIR}/m6_spectral_mlp_vpinn_data.npz', 'M6: Spectral+MLP VPINN'),
}
for key, (fname, label) in specs.items():
    d = np.load(fname)
    data[key] = {
        'U': d['U'], 'V': d['V'], 'P': d['P'],
        'xx': d['xx'], 'yy': d['yy'],
        'label': label,
        'loss': float(d['final_loss']),
        'time_h': float(d.get('total_time', d.get('adam_time', 0) + d.get('lbfgs_time', 0))) / 3600,
        'adam_t': float(d.get('adam_time', 0)) / 3600,
        'lbfgs_t': float(d.get('lbfgs_time', 0)) / 3600,
    }

# Reference = M2
ref = data['M2']
for key in ['M1', 'M3', 'M4', 'M5', 'M6']:
    d = data[key]
    d['eps_u'] = np.linalg.norm(d['U'] - ref['U']) / np.linalg.norm(ref['U'])
    d['eps_v'] = np.linalg.norm(d['V'] - ref['V']) / np.linalg.norm(ref['V'])
    d['eps_p'] = np.linalg.norm(d['P'] - ref['P']) / (np.linalg.norm(ref['P']) + 1e-14)
data['M2']['eps_u'] = 0.0
data['M2']['eps_v'] = 0.0
data['M2']['eps_p'] = 0.0

# Load loss histories
losses = {}
loss_files = {
    'M1': f'{DATA_DIR}/cavity2D_m1_losses.npz',
    'M2': f'{DATA_DIR}/cavity2D_m2_losses.npz',
    'M3': f'{DATA_DIR}/cavity2D_m4_losses.npz',
    'M4': f'{DATA_DIR}/cavity2D_m3_losses.npz',
    'M5': f'{DATA_DIR}/cavity2D_m5_losses.npz',
    'M6': f'{DATA_DIR}/m6_spectral_mlp_vpinn_losses.npz',
}
for key, fname in loss_files.items():
    try:
        ld = np.load(fname)
        losses[key] = {k: ld[k] for k in ld.files}
    except:
        losses[key] = {}

COLORS = {'M1': '#e41a1c', 'M2': '#377eb8', 'M3': '#4daf4a', 'M4': '#ff7f00', 'M5': '#984ea3', 'M6': '#a65628'}
MARKERS = {'M1': 'o', 'M2': 's', 'M3': 'D', 'M4': '^', 'M5': 'v', 'M6': 'p'}

ALL_KEYS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']

# ── Fig 4: Streamlines (3×2 layout with 6 methods) ─────────────────────────
print("Generating Fig 4 — Streamlines ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, key in enumerate(ALL_KEYS):
    ax = axes[idx]
    d = data[key]
    xx2d, yy2d = d['xx'], d['yy']
    x1d = xx2d[:, 0] if xx2d.ndim == 2 else xx2d
    y1d = yy2d[0, :] if yy2d.ndim == 2 else yy2d
    U, V = d['U'], d['V']
    speed = np.sqrt(U**2 + V**2)
    strm = ax.streamplot(x1d, y1d, U.T, V.T, color=speed.T,
                         cmap='viridis', linewidth=0.8, density=1.5,
                         arrowsize=0.8)
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    cbar = plt.colorbar(strm.lines, ax=ax, label='|u|', fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

fig.suptitle(r'Streamlines colored by $|\mathbf{u}|$, Re = 100', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 4. Cavity streamlines.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 5: Centerline profiles ─────────────────────────────────────────────
print("Generating Fig 5 — Centerline profiles ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for key in ALL_KEYS:
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    U, V = d['U'], d['V']
    ix = np.argmin(np.abs(x1d))
    axes[0].plot(U[ix, :], y1d, color=COLORS[key], marker=MARKERS[key],
                 markevery=10, label=d['label'], linewidth=1.8)
    iy = np.argmin(np.abs(y1d))
    axes[1].plot(x1d, V[:, iy], color=COLORS[key], marker=MARKERS[key],
                 markevery=10, label=d['label'], linewidth=1.8)

axes[0].set_xlabel(r'$u$-velocity', fontsize=12)
axes[0].set_ylabel(r'$y$', fontsize=12)
axes[0].set_title(r'$u(x\approx 0, y)$ — vertical centerline', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel(r'$x$', fontsize=12)
axes[1].set_ylabel(r'$v$-velocity', fontsize=12)
axes[1].set_title(r'$v(x, y\approx 0)$ — horizontal centerline', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle('Centerline velocity profiles, Re = 100', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 5. Cavity centerline profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 6: u-velocity contours (3×2) ───────────────────────────────────────
print("Generating Fig 6 — U-velocity contours ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
vmin = min(data[k]['U'].min() for k in data)
vmax = max(data[k]['U'].max() for k in data)
levels = np.linspace(vmin, vmax, 21)

for idx, key in enumerate(ALL_KEYS):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    cf = ax.contourf(x1d, y1d, d['U'].T, levels=levels, cmap='RdBu_r')
    ax.contour(x1d, y1d, d['U'].T, levels=levels[::4], colors='k', linewidths=0.4, alpha=0.4)
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='u', fraction=0.046)

fig.suptitle(r'$u$-velocity field, Re = 100', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 6. Cavity U-velocity contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 7: v-velocity contours ─────────────────────────────────────────────
print("Generating Fig 7 — V-velocity contours ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
vmin_v = min(data[k]['V'].min() for k in data)
vmax_v = max(data[k]['V'].max() for k in data)
vlevels = np.linspace(vmin_v, vmax_v, 21)

for idx, key in enumerate(ALL_KEYS):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    cf = ax.contourf(x1d, y1d, d['V'].T, levels=vlevels, cmap='RdBu_r')
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='v', fraction=0.046)

fig.suptitle(r'$v$-velocity field, Re = 100', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 7. Cavity V-velocity contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 8: Pressure fields (3×2) ───────────────────────────────────────────
print("Generating Fig 8 — Pressure fields ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
pall = np.concatenate([data[k]['P'].ravel() for k in data])
plevels = np.linspace(pall.min(), pall.max(), 21)

for idx, key in enumerate(ALL_KEYS):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    cf = ax.contourf(x1d, y1d, d['P'].T, levels=plevels, cmap='seismic')
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='p', fraction=0.046)

fig.suptitle('Pressure field $p$, Re = 100', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 8. Cavity pressure contours.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 9: Loss convergence ────────────────────────────────────────────────
print("Generating Fig 9 — Loss convergence ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Adam phase
for key, ld in losses.items():
    if 'adam_losses' in ld:
        al = ld['adam_losses']
        axes[0].semilogy(np.arange(len(al)) + 1, al,
                         color=COLORS[key], label=data[key]['label'], linewidth=1.5)
axes[0].set_xlabel('Adam epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Phase 1: Adam optimizer', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, which='both', alpha=0.3)

# L-BFGS phase
for key, ld in losses.items():
    if 'lbfgs_losses' in ld:
        ll = ld['lbfgs_losses']
        ll = ll[ll > 0]
        axes[1].semilogy(np.arange(len(ll)) + 1, ll,
                         color=COLORS[key], label=data[key]['label'], linewidth=1.5)
axes[1].set_xlabel('L-BFGS step', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Phase 2: L-BFGS optimizer', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, which='both', alpha=0.3)

# Mark final losses
for key in ALL_KEYS:
    axes[1].axhline(data[key]['loss'], color=COLORS[key], linestyle=':', alpha=0.5, linewidth=1)

fig.suptitle('Training loss convergence', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 9. Cavity loss convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 10: Relative errors bar chart ──────────────────────────────────────
print("Generating Fig 10 — Error comparison ...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

keys_cmp = ['M1', 'M3', 'M4', 'M5', 'M6']
labels_cmp = [data[k]['label'] for k in keys_cmp]
x = np.arange(len(keys_cmp))
bar_colors = [COLORS[k] for k in keys_cmp]

for ax_idx, (field, title) in enumerate([('eps_u', r'$\epsilon_u$ (u-velocity)'),
                                          ('eps_v', r'$\epsilon_v$ (v-velocity)'),
                                          ('eps_p', r'$\epsilon_p$ (pressure)')]):
    ax = axes[ax_idx]
    vals = [data[k][field] for k in keys_cmp]
    bars = ax.bar(x, vals, color=bar_colors, alpha=0.85, width=0.6)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([k for k in keys_cmp], fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Relative L2 error', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.15,
                f'{val:.2e}', ha='center', va='bottom', fontsize=8, rotation=0)

fig.suptitle('Relative $\\ell_2$ errors vs. M2 (Spectral LSTSQ) reference', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 10. Cavity error comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 11: Pointwise error fields |u - u_ref| ────────────────────────────
print("Generating Fig 11 — Error fields ...")
keys_pw = ['M1', 'M3', 'M4', 'M5', 'M6']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, key in enumerate(keys_pw):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    err_u = np.abs(d['U'] - ref['U'])
    cf = ax.contourf(x1d, y1d, err_u.T, levels=21, cmap='hot_r')
    ax.set_title(f"{d['label']}\n" + r"$|u - u_{\rm ref}|$", fontsize=11, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='|error|', fraction=0.046)

axes[5].set_visible(False)  # 5 methods shown, last panel hidden
fig.suptitle(r'Pointwise $|u - u_{\rm ref}|$ error (ref = M2 Spectral LSTSQ)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 11. Cavity error fields.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 12: Vorticity fields ───────────────────────────────────────────────
print("Generating Fig 12 — Vorticity ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, key in enumerate(ALL_KEYS):
    ax = axes[idx]
    d = data[key]
    x1d = d['xx'][:, 0]; y1d = d['yy'][0, :]
    U, V = d['U'], d['V']
    dx = x1d[1] - x1d[0]
    dy = y1d[1] - y1d[0]
    dVdx = np.gradient(V, dx, axis=0)
    dUdy = np.gradient(U, dy, axis=1)
    omega = dVdx - dUdy
    vmax_w = np.percentile(np.abs(omega), 99)
    cf = ax.contourf(x1d, y1d, omega.T, levels=np.linspace(-vmax_w, vmax_w, 21),
                     cmap='RdBu_r', extend='both')
    ax.set_title(d['label'], fontsize=12, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label=r'$\omega$', fraction=0.046)

fig.suptitle(r'Vorticity $\omega = \partial v/\partial x - \partial u/\partial y$, Re = 100',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 12. Cavity vorticity comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Fig 13: Efficiency chart ───────────────────────────────────────────────
print("Generating Fig 13 — Efficiency scatter ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels_all = [data[k]['label'] for k in ALL_KEYS]
x = np.arange(len(ALL_KEYS))

# Training time (stacked: Adam + L-BFGS)
adam_times = [data[k]['adam_t'] for k in ALL_KEYS]
lbfgs_times = [data[k]['lbfgs_t'] for k in ALL_KEYS]
axes[0].bar(x, adam_times, color=[COLORS[k] for k in ALL_KEYS], alpha=0.6,
            label='Adam', width=0.6)
axes[0].bar(x, lbfgs_times, bottom=adam_times,
            color=[COLORS[k] for k in ALL_KEYS], alpha=0.95, label='L-BFGS', width=0.6,
            hatch='//')
axes[0].set_xticks(x)
axes[0].set_xticklabels([k for k in ALL_KEYS], fontsize=11)
axes[0].set_ylabel('Training time (h)', fontsize=12)
axes[0].set_title('Computational cost', fontsize=12)
axes[0].legend(['Adam phase', 'L-BFGS phase'], fontsize=10)
axes[0].grid(True, axis='y', alpha=0.3)
for xi, (at, lt) in enumerate(zip(adam_times, lbfgs_times)):
    total = at + lt
    axes[0].text(xi, total + 0.05, f'{total:.2f}h', ha='center', fontsize=8)

# Final loss
losses_all = [data[k]['loss'] for k in ALL_KEYS]
bars3 = axes[1].bar(x, losses_all, color=[COLORS[k] for k in ALL_KEYS], alpha=0.85, width=0.6)
axes[1].set_yscale('log')
axes[1].set_xticks(x)
axes[1].set_xticklabels([k for k in ALL_KEYS], fontsize=11)
axes[1].set_ylabel('Final training loss', fontsize=12)
axes[1].set_title('Solution accuracy', fontsize=12)
axes[1].grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars3, losses_all):
    axes[1].text(bar.get_x() + bar.get_width()/2, val * 1.5,
                 f'{val:.2e}', ha='center', va='bottom', fontsize=7.5)

fig.suptitle('Efficiency and accuracy comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/Fig 13. Cavity efficiency scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  done")

# ── Print summary ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("COMPLETE RESULTS SUMMARY (v7 with M6)")
print("="*80)
print(f"{'Method':<25} {'Loss':>12} {'eps_u':>10} {'eps_v':>10} {'eps_p':>10} {'Time(h)':>8}")
print("-"*80)
for key in ALL_KEYS:
    d = data[key]
    eu = d['eps_u']; ev = d['eps_v']; ep = d['eps_p']
    print(f"{d['label']:<25} {d['loss']:>12.3e} {eu:>10.3e} {ev:>10.3e} {ep:>10.3e} {d['time_h']:>8.3f}")
print("="*80)
print("All figures saved to paper_figures/")
