import os
import csv

import numpy as np
import matplotlib.pyplot as plt

# ── Where the CSVs written by the run scripts live ────────────────────────────
# Error vs mesh size h (A / B scripts)
CSV_FILES = {
    "A":  {"GFEM": "plots_A/errors.csv",  "Stabilized GFEM": "plots_As/errors.csv"},
    "B":  {"GFEM": "plots_B/errors.csv",  "Stabilized GFEM": "plots_Bs/errors.csv"},
}

# Error vs CFL number (run_convection_cfl.py)
CFL_CSV = "plots_for_1-3/errors.csv"

# Nominal mesh sizes, keyed by node count (from your notes). If a node count is
# not listed here, the h column from the CSV is used instead.
NOMINAL_H = {
    1594: 0.05,
    649:  0.08,
    423:  0.10,
    331:  0.12,
}

OUT_DIR = "errorplots"
os.makedirs(OUT_DIR, exist_ok=True)


# ══ Error vs h ════════════════════════════════════════════════════════════════
def load_csv(path):
    """Return (h, error) arrays sorted by h. Repeated runs on the same mesh
    (same node count) keep only the last one. Returns None if no file/rows."""
    if not os.path.exists(path):
        return None

    latest = {}  # n_nodes -> (h, error)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n_nodes = int(row["n_nodes"])
            h = NOMINAL_H.get(n_nodes, float(row["h"]))
            latest[n_nodes] = (h, float(row["L2_error"]))

    if not latest:
        return None

    pairs = sorted(latest.values())
    h = np.array([a for a, _ in pairs])
    err = np.array([b for _, b in pairs])
    return h, err


def plotfig(series, init_cond):
    """series: dict label -> (h, err). Saves errorplots/error_<cond>_vs_as.png"""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, (h, err) in series.items():
        ax.loglog(h, err, "o-", label=label)
        if len(h) >= 2:
            rate = np.polyfit(np.log(h), np.log(err), 1)[0]
            print(f"  {init_cond} / {label}: observed order ~ {rate:.2f}")

    ax.set_xlabel("Characteristic mesh size h")
    ax.set_ylabel(r"$e_h$")
    ax.set_title(f"Initial condition {init_cond}")
    ax.set_ylim(1e-4, 1)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(OUT_DIR, f"error_{init_cond}_vs_as.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


for cond, files in CSV_FILES.items():
    series = {}
    for label, path in files.items():
        data = load_csv(path)
        if data is None:
            print(f"[{cond}] no data for '{label}' ({path}), skipping")
        else:
            series[label] = data

    if series:
        plotfig(series, cond)
    else:
        print(f"[{cond}] nothing to plot")


# ══ Error vs CFL ══════════════════════════════════════════════════════════════
def load_cfl_csv(path):
    """Read the CSV from run_convection_cfl.py.

    Returns {label: (cfl array, error array)} with one curve per
    (integrator, node count[, max_time]), sorted by CFL. Repeated runs with the
    same settings keep only the last one. Runs whose error is not finite
    (blown up to nan/inf) are left out and counted. Returns None if no data.
    """
    if not os.path.exists(path):
        return None

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    several_times = len({float(r["max_time"]) for r in rows}) > 1

    latest = {}   # (label, cfl) -> error
    dropped = 0
    for r in rows:
        err = float(r["L2_error"])
        if not np.isfinite(err):
            dropped += 1
            continue
        label = f'{r["integration"]}, {int(r["n_nodes"])} nodes'
        if several_times:
            label += f', t={float(r["max_time"]):g}'
        latest[(label, float(r["CFL"]))] = err

    if dropped:
        print(f"  [CFL] left out {dropped} run(s) with non-finite error (blow-up)")

    curves = {}
    for (label, cfl), err in latest.items():
        curves.setdefault(label, []).append((cfl, err))

    out = {}
    for label, pairs in curves.items():
        pairs.sort()
        out[label] = (np.array([a for a, _ in pairs]),
                      np.array([b for _, b in pairs]))
    return out or None


def plot_cfl(curves):
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, (cfl, err) in sorted(curves.items()):
        ax.loglog(cfl, err, "o-", label=label)

    ax.set_xlabel("CFL number")
    ax.set_ylabel(r"$e_h$")
    ax.set_title("Error as a function of CFL number")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(OUT_DIR, "error_vs_CFL.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


cfl_curves = load_cfl_csv(CFL_CSV)
if cfl_curves is None:
    print(f"[CFL] no data in {CFL_CSV}, skipping")
else:
    plot_cfl(cfl_curves)