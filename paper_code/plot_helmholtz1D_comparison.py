"""
Generate 1D Helmholtz 6-method comparison figures for the paper.

Methods (1D mapping to M1-M6 framework):
  M1: PINN (strong-form least-squares on MLP trial)
  M2: Spectral-LSTSQ (direct spectral Galerkin solve)
  M3: SEM-LSTSQ (piecewise collocation, strong-form)
  M4: Spectral-VPINN (global Legendre test functions)
  M5: SEM-VPINN (piecewise Legendre test functions, weak-form)
  M6: Spectral+MLP VPINN (spectral trial + MLP test, Galerkin)

Outputs:
  paper_figures/Fig 1. Helmholtz solution comparison.png
  paper_figures/Fig 2. Helmholtz pointwise error.png
  paper_figures/Fig 3. Helmholtz error and runtime comparison.png
"""
# ruff: noqa: E402
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# Load all results
# ============================================================
FIG_DIR = "/Users/gyh/WorkBuddy/20260415163831/paper_figures"
DATA_DIR = "/Users/gyh/WorkBuddy/20260415163831/paper_data"

# Map M-numbers to data files
files = {
    "M1: PINN":                 f"{DATA_DIR}/helmholtz1D_lstsq_pinn_results.npz",
    "M2: Spectral-LSTSQ":       f"{DATA_DIR}/helmholtz1D_spectral_results.npz",
    "M3: SEM-LSTSQ":            f"{DATA_DIR}/helmholtz1D_sem_lstsq_results.npz",
    "M4: Spectral-VPINN":       f"{DATA_DIR}/helmholtz1D_vpinn_results.npz",
    "M5: SEM-VPINN":            f"{DATA_DIR}/helmholtz1D_sem_vpinn_results.npz",
    "M6: Spectral+MLP VPINN":   f"{DATA_DIR}/helmholtz1D_m6_results.npz",
}

data = {}
for name, path in files.items():
    try:
        d = np.load(path, allow_pickle=True)
        data[name] = d
        print(f"Loaded {name}: keys={list(d.keys())}")
    except FileNotFoundError:
        print(f"WARNING: {name} data not found at {path}")

# Use common evaluation grid
x_eval = None
for name in data:
    if "xj" in data[name]:
        x_eval = data[name]["xj"]
        break
if x_eval is None:
    x_eval = np.linspace(-1, 1, 1000)

x_eval = np.array(x_eval).reshape(-1)

# ============================================================
# Extract u_pred and ue for each method
# ============================================================
def get_pred_and_exact(d):
    """Extract predicted and exact solution from a result dict."""
    if "uj" in d:
        u_pred = np.array(d["uj"]).reshape(-1)
    elif "w_pred" in d:
        u_pred = np.array(d["w_pred"]).reshape(-1)
    else:
        raise KeyError(f"Cannot find predicted solution in keys: {list(d.keys())}")
    if "uej" in d:
        ue = np.array(d["uej"]).reshape(-1)
    else:
        raise KeyError(f"Cannot find exact solution in keys: {list(d.keys())}")
    return u_pred, ue

method_data = {}
for name in data:
    u_pred, ue = get_pred_and_exact(data[name])
    if len(u_pred) != len(x_eval):
        from scipy import interpolate
        x_orig = data[name]["xj"].reshape(-1) if "xj" in data[name] else np.linspace(-1, 1, len(u_pred))
        u_pred = interpolate.interp1d(x_orig, u_pred, kind="linear")(x_eval)
        ue = interpolate.interp1d(x_orig, ue, kind="linear")(x_eval)
    method_data[name] = {"u_pred": u_pred, "ue": ue}

# M1-M6 display order
METHOD_ORDER = [
    "M1: PINN",
    "M2: Spectral-LSTSQ",
    "M3: SEM-LSTSQ",
    "M4: Spectral-VPINN",
    "M5: SEM-VPINN",
    "M6: Spectral+MLP VPINN",
]

# Visual properties
colors = {
    "M1: PINN":                 "C0",
    "M2: Spectral-LSTSQ":       "k",
    "M3: SEM-LSTSQ":            "C2",
    "M4: Spectral-VPINN":       "C3",
    "M5: SEM-VPINN":            "C1",
    "M6: Spectral+MLP VPINN":   "C4",
}
linestyles = {
    "M1: PINN":                 ":",
    "M2: Spectral-LSTSQ":       "-",
    "M3: SEM-LSTSQ":            (0, (3, 1, 1, 1)),
    "M4: Spectral-VPINN":       "--",
    "M5: SEM-VPINN":            "-.",
    "M6: Spectral+MLP VPINN":   (0, (5, 2)),
}
linewidths = {
    "M1: PINN":                 2.0,
    "M2: Spectral-LSTSQ":       2.5,
    "M3: SEM-LSTSQ":            2.0,
    "M4: Spectral-VPINN":       2.0,
    "M5: SEM-VPINN":            2.0,
    "M6: Spectral+MLP VPINN":   2.5,
}

# ============================================================
# Figure 1: Solution comparison (all 6 methods overlaid)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot exact solution
ue_ref = method_data["M2: Spectral-LSTSQ"]["ue"]
ax.plot(x_eval, ue_ref, "k-", lw=3, alpha=0.3, label="Exact (reference)")

for name in METHOD_ORDER:
    d = method_data[name]
    ax.plot(x_eval, d["u_pred"],
            color=colors[name],
            ls=linestyles[name],
            lw=linewidths[name],
            label=name)

ax.set_xlabel("x", fontsize=13)
ax.set_ylabel("u(x)", fontsize=13)
ax.set_title("1D Helmholtz: Solution Comparison (M1--M6)", fontsize=14)
ax.legend(fontsize=9, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Fig 1. Helmholtz solution comparison.png", dpi=150)
print("Saved: Fig 1. Helmholtz solution comparison.png")
plt.close()

# ============================================================
# Figure 2: Pointwise error comparison
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for name in METHOD_ORDER:
    d = method_data[name]
    err = np.abs(d["u_pred"] - d["ue"])
    ax.semilogy(x_eval, err + 1e-20,
                color=colors[name],
                ls=linestyles[name],
                lw=1.8,
                label=name)

ax.set_xlabel("x", fontsize=13)
ax.set_ylabel("|error| (log scale)", fontsize=13)
ax.set_title("1D Helmholtz: Pointwise Error Comparison (M1--M6)", fontsize=14)
ax.legend(fontsize=9, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Fig 2. Helmholtz pointwise error.png", dpi=150)
print("Saved: Fig 2. Helmholtz pointwise error.png")
plt.close()

# ============================================================
# Figure 3: Bar chart — L2 error and runtime
# ============================================================
methods_short = ["M1\nPINN", "M2\nSpectral-\nLSTSQ", "M3\nSEM-\nLSTSQ",
                 "M4\nSpectral-\nVPINN", "M5\nSEM-\nVPINN",
                 "M6\nSpectral+\nMLP VPINN"]
l2_errors = []
runtimes = []
for name in METHOD_ORDER:
    d = data[name]
    if "l2_error" in d:
        l2_errors.append(float(d["l2_error"]))
    elif "error" in d:
        l2_errors.append(float(d["error"]))
    else:
        l2_errors.append(np.nan)
    if "total_time" in d:
        runtimes.append(float(d["total_time"]))
    elif "time" in d:
        runtimes.append(float(d["time"]))
    elif "time_adam" in d and "time_lbfgs" in d:
        runtimes.append(float(d["time_adam"]) + float(d["time_lbfgs"]))
    else:
        runtimes.append(np.nan)

x = np.arange(len(methods_short))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
bars1 = ax1.bar(x - width/2, l2_errors, width, label="L2 error", color="C0", alpha=0.8)
ax1.set_ylabel("L2 error (log scale)", fontsize=12, color="C0")
ax1.set_yscale("log")
ax1.tick_params(axis="y", labelcolor="C0")
ax1.set_xticks(x)
ax1.set_xticklabels(methods_short, fontsize=8)
ax1.grid(True, alpha=0.3, which="both", axis="y")

for bar, val in zip(bars1, l2_errors):
    height = bar.get_height()
    ax1.annotate(f"{val:.2e}",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha="center", fontsize=8)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, runtimes, width, label="Runtime (s)", color="C3", alpha=0.8)
ax2.set_ylabel("Runtime (seconds)", fontsize=12, color="C3")
ax2.tick_params(axis="y", labelcolor="C3")
ax2.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars2, runtimes):
    height = bar.get_height()
    ax2.annotate(f"{val:.1f}s",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha="center", fontsize=8)

ax1.set_title("1D Helmholtz: L2 Error and Runtime Comparison (M1--M6)", fontsize=14, pad=15)
fig.tight_layout()
plt.savefig(f"{FIG_DIR}/Fig 3. Helmholtz error and runtime comparison.png", dpi=150)
print("Saved: Fig 3. Helmholtz error and runtime comparison.png")
plt.close()

print("\nAll comparison figures saved to paper_figures/")
print(f"L2 errors: {[f'{e:.2e}' for e in l2_errors]}")
print(f"Runtimes:  {[f'{t:.1f}s' for t in runtimes]}")
