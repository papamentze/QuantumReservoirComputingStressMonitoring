#!/usr/bin/env python
"""
fig_mechanism.py - main-text figure and exact statistics for the
recurrence x noise ablation.

Input : mech_auroc.csv written by qrc_mechanism.py
        (columns: dataset, modality, draw, leak, kappa, subject, auroc)
Output: fig_noise_mechanism.png / .tiff / .pdf
        mech_gap_by_draw.csv   recurrent-minus-memoryless gap per reservoir draw and kappa
        mech_gap_summary.csv   per-cell and pooled statistics quoted in the paper

The memoryless reservoir (leak = 1) is the reference. Its AUROC does not depend on
kappa (checked and printed), so the gap
    Delta(kappa) = AUROC(leak = 0.15, kappa) - AUROC(leak = 1, kappa)
isolates what recurrence contributes at each noise level. The simulation is a
deterministic density-matrix computation, so any change in Delta across kappa is
caused by the noise channel, not by sampling noise.

Usage : python fig_mechanism.py --out Output/V2/mechanism
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

CELLS = [("wesad", "hrv"), ("wesad", "eda"), ("wesad", "resp"),
         ("road", "hrv"), ("road", "eda")]
NAMES = {("wesad", "hrv"): "WESAD-HRV", ("wesad", "eda"): "WESAD-EDA",
         ("wesad", "resp"): "WESAD-RESP", ("road", "hrv"): "ROAD-HRV",
         ("road", "eda"): "ROAD-EDA"}
REC, MEM = 0.15, 1.0
C_MEAN, C_BAND, C_DRAW = "#0072B2", "#56B4E9", "#8c8c8c"   # Okabe-Ito, colour-blind safe
C_K0, C_KHI = "#BBBBBB", "#0072B2"


def load(path):
    df = pd.read_csv(path)
    df["leak"] = df["leak"].round(4)
    df["kappa"] = df["kappa"].round(4)
    # subject-mean AUROC per reservoir draw, the same aggregation as mech_summary.csv
    return (df.groupby(["dataset", "modality", "draw", "leak", "kappa"], as_index=False)
              ["auroc"].mean())


def check_invariance(dm):
    mem = dm[dm.leak == MEM]
    span = mem.groupby(["dataset", "modality", "draw"])["auroc"].agg(lambda s: s.max() - s.min())
    print("memoryless reservoir: largest AUROC change across kappa (worst draw per cell)")
    for (d, m), s in span.groupby(level=[0, 1]):
        print(f"  {NAMES[(d, m)]:<11s} {s.max():.2e}")
    return float(span.max())


def gaps(dm):
    key = ["dataset", "modality", "draw", "kappa"]
    rec = dm[dm.leak == REC].set_index(key)["auroc"]
    mem = dm[dm.leak == MEM].set_index(key)["auroc"]
    out = pd.concat({"rec": rec, "mem": mem}, axis=1).dropna().reset_index()
    out["gap"] = out["rec"] - out["mem"]
    return out


def safe_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a - b, 0):
        return np.nan
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return np.nan


def summarise(g, k_hi):
    """Per-cell statistics (primary) plus a pooled row (descriptive: reservoirs
    within a cell share subjects, so the pooled p-value is not independent)."""
    groups = [(NAMES[c], g[(g.dataset == c[0]) & (g.modality == c[1])]) for c in CELLS]
    groups.append(("pooled (all reservoirs)", g))
    rows = []
    for name, s in groups:
        idx = ["dataset", "modality", "draw"]
        gk = s.pivot_table(index=idx, columns="kappa", values="gap")
        rk = s.pivot_table(index=idx, columns="kappa", values="rec")
        g0, gh = gk[0.0], gk[k_hi]
        a0, ah = g0.abs(), gh.abs()
        rows.append(dict(
            cell=name,
            n_reservoirs=len(gk),
            gap_k0=g0.mean(),                       # signed: <0 means recurrence costs AUROC
            gap_khi=gh.mean(),
            absgap_k0=a0.mean(),                    # distance to the memoryless twin
            absgap_khi=ah.mean(),
            n_closer=int((ah < a0 - 1e-9).sum()),   # reservoirs pulled toward memoryless
            p_closer=safe_wilcoxon(a0, ah),
            n_rec_above_mem_k0=int((g0 > 0).sum()),
            n_rec_above_mem_khi=int((gh > 0).sum()),
            noise_effect=(rk[k_hi] - rk[0.0]).mean(),
            n_noise_gain=int((rk[k_hi] > rk[0.0]).sum()),
        ))
    return pd.DataFrame(rows)


def boot_ci(x, rng, n=5000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.array([x.mean(), x.mean()])
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])

def figure(g, summ, k_hi, out, width):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.titlesize": 6.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(width, width * 0.8), layout="constrained")
    gap_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    lo, hi = g["gap"].min(), g["gap"].max()
    pad = 0.08 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    kappas = np.sort(g["kappa"].unique())
    summ_i = summ.set_index("cell")

    for i, (cell, ax) in enumerate(zip(CELLS, gap_axes)):
        s = g[(g.dataset == cell[0]) & (g.modality == cell[1])]
        for _, d in s.groupby("draw"):
            d = d.sort_values("kappa")
            ax.plot(d.kappa, d.gap, color=C_DRAW, lw=0.5, alpha=0.45, zorder=1)
        piv = s.pivot_table(index="draw", columns="kappa", values="gap")[kappas]
        ci = np.array([boot_ci(piv[k].values, rng) for k in kappas])
        ax.fill_between(kappas, ci[:, 0], ci[:, 1], color=C_BAND, alpha=0.3, lw=0, zorder=2)
        ax.plot(kappas, piv.mean().values, color=C_MEAN, lw=1.4, marker="o", ms=2.5, zorder=3)
        ax.axhline(0, color="k", lw=0.8, ls="--", zorder=2)
        ax.set_ylim(ylim)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("noise scale κ")
        ax.set_title(f"({'abcde'[i]}) {NAMES[cell]}", loc="left", fontweight="bold")
        r = summ_i.loc[NAMES[cell]]
        ax.text(0.97, 0.04,
                f"|Δ| {r.absgap_k0:.3f} → {r.absgap_khi:.3f}\n"
                f"closer in {int(r.n_closer)}/{int(r.n_reservoirs)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9))
    for ax in (axes[0, 1], axes[0, 2], axes[1, 1]):
        ax.tick_params(labelleft=False)
    fig.supylabel("Δ AUROC (recurrent − memoryless)", fontsize=7)

    ax = axes[1, 2]
    x, w = np.arange(len(CELLS)), 0.38
    m0, mh, e0, eh = [], [], [], []
    for cell in CELLS:
        s = g[(g.dataset == cell[0]) & (g.modality == cell[1])]
        piv = s.pivot_table(index="draw", columns="kappa", values="gap").abs()
        m0.append(piv[0.0].mean()); mh.append(piv[k_hi].mean())
        e0.append(piv[0.0].std(ddof=1)); eh.append(piv[k_hi].std(ddof=1))
    ax.bar(x - w / 2, m0, w, yerr=e0, color=C_K0, capsize=1.5, error_kw=dict(lw=0.6), label="κ = 0")
    ax.bar(x + w / 2, mh, w, yerr=eh, color=C_KHI, capsize=1.5, error_kw=dict(lw=0.6), label=f"κ = {k_hi:g}")
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[c] for c in CELLS], rotation=35, ha="right",
                       rotation_mode="anchor", fontsize=6)
    ax.set_ylabel("mean |Δ AUROC|")
    ax.set_title("(f) κ = 0 vs. 3", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=6)

    handles = [
        Line2D([], [], color=C_DRAW, lw=0.7, alpha=0.7, label="single reservoir draw"),
        Line2D([], [], color=C_MEAN, lw=1.4, marker="o", ms=2.5, label="mean over draws"),
        Patch(color=C_BAND, alpha=0.3, label="95% bootstrap CI"),
        Line2D([], [], color="k", lw=0.8, ls="--", label="memoryless (ε = 1), κ-invariant"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)

    stem = out / "fig_noise_mechanism"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig) 

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="Output/V2/mechanism")
    ap.add_argument("--kappa-hi", type=float, default=3.0)
    ap.add_argument("--width", type=float, default=4.1,
                    help="figure width in inches (4.1 in = 0.85 x llncs text width)")
    a = ap.parse_args()
    out = Path(a.out)
    if "FINAL" in out.parts:
        raise SystemExit("refusing to write inside Output/FINAL")

    dm = load(out / "mech_auroc.csv")
    check_invariance(dm)
    g = gaps(dm)
    k_hi = round(a.kappa_hi, 4)
    if k_hi not in set(g.kappa):
        raise SystemExit(f"kappa {k_hi} not in grid {sorted(g.kappa.unique())}")

    g.to_csv(out / "mech_gap_by_draw.csv", index=False)
    summ = summarise(g, k_hi)
    summ.to_csv(out / "mech_gap_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(summ.to_string(index=False))
    figure(g, summ, k_hi, out, a.width)
    print(f"wrote fig_noise_mechanism.png/.tiff/.pdf, mech_gap_by_draw.csv, "
          f"mech_gap_summary.csv to {out}")


if __name__ == "__main__":
    main() 