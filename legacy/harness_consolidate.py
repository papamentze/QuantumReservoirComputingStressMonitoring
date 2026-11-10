"""
================================================================================
 harness_consolidate.py -- pool qrc_harness.py per-fold CSVs across all cells
 (dataset x modality) into the paper's benchmark-level verdict.

 Reads every Output/harness/<dataset>/<mod>/harness_perfold_<mod>.csv and produces:
   * master per-cell subject-mean table (auroc/auprc/ewma per method)
   * per-cell TOST(QRC vs ESN)                          -> local equivalence
   * BENCHMARK equivalence two honest ways:
       (A) pooled diffs  : every (subject,cell) qrc-esn diff as one unit
                           -> maximum power, but diffs from the SAME subject across
                              modalities are NOT independent (state this as a limit).
       (B) subject-avg   : average each subject's qrc-esn diff across their cells
                           first -> one independent unit per (dataset,subject).
                           Fewer units, but clean independence. This is the one to
                           lead with; (A) is a sensitivity check.
   * paired QRC-QSVM / QRC-VQC diffs if those methods are present.

 Nothing is selected on a metric here; it only aggregates what the harness wrote.
 USAGE
   python harness_consolidate.py --root Output/harness --metric auroc --out Output/harness
================================================================================
"""

from __future__ import annotations
import os, sys, glob, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qrc_base import tost_paired, bootstrap_ci
from scipy import stats

MARGINS = (0.01, 0.02, 0.03, 0.05)

def _load_all(root):
    rows = []
    for p in glob.glob(os.path.join(root, "*", "*", "harness_perfold_*.csv")):
        parts = p.replace("\\", "/").split("/")
        dataset, mod = parts[-3], parts[-2]
        df = pd.read_csv(p)
        df["dataset"], df["modality"] = dataset, mod
        rows.append(df)
    if not rows:
        raise SystemExit(f"[error] no harness_perfold_*.csv under {root}")
    return pd.concat(rows, ignore_index=True)

def _subject_means(pf, metric):
    return (pf.groupby(["dataset", "modality", "method", "subject"])[metric]
              .mean().reset_index())

def master_table(pf, metric):
    sm = _subject_means(pf, metric)
    tbl = (sm.groupby(["dataset", "modality", "method"])[metric]
             .agg(["mean", "count"]).reset_index()
             .pivot_table(index=["dataset", "modality"], columns="method", values="mean"))
    return tbl.round(3)

def per_cell_tost(pf, metric, a="qrc", b="esn"):
    out = []
    for (ds, mod), g in pf.groupby(["dataset", "modality"]):
        if not {a, b} <= set(g.method.unique()):
            continue
        t = tost_paired(g, metric, [(a, b)], list(MARGINS))
        if len(t):
            t.insert(0, "dataset", ds); t.insert(1, "modality", mod)
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def _diffs(pf, metric, a, b):
    """One qrc-b diff per (dataset,modality,subject)."""
    sm = _subject_means(pf, metric)
    xa = sm[sm.method == a].set_index(["dataset", "modality", "subject"])[metric]
    xb = sm[sm.method == b].set_index(["dataset", "modality", "subject"])[metric]
    j = pd.concat([xa, xb], axis=1, keys=["a", "b"]).dropna().reset_index()
    j["diff"] = j["a"] - j["b"]
    return j

def _tost_on_diffs(d, label, margins=MARGINS):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    n = len(d); mean = d.mean(); se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    rows = []
    for delta in margins:
        if not np.isfinite(se) or se == 0:
            p, eq = np.nan, False
        else:
            p_lo = stats.t.sf((mean + delta) / se, n - 1)
            p_hi = stats.t.cdf((mean - delta) / se, n - 1)
            p = max(p_lo, p_hi); eq = bool(p < 0.05)
        rows.append(dict(test=label, n=n, margin_pp=delta * 100, mean_diff=mean,
                         se=se, p_tost=p, equivalent=eq))
    return pd.DataFrame(rows)

def benchmark_equivalence(pf, metric, a="qrc", b="esn"):
    j = _diffs(pf, metric, a, b)
    # (A) pooled: every (subject,cell) diff (max power, mild non-independence)
    A = _tost_on_diffs(j["diff"].values, f"POOLED {a}-{b} [{metric}] (non-indep)")
    # (B) subject-averaged within dataset (clean independence)
    savg = (j.groupby(["dataset", "subject"])["diff"].mean().values)
    B = _tost_on_diffs(savg, f"SUBJ-AVG {a}-{b} [{metric}] (independent)")
    return pd.concat([B, A], ignore_index=True)

def best_quantum(pf, metric):
    out = []
    for comp in ("qsvm", "vqc", "qlstm", "qgru", "esn", "memoryless"):
        j = _diffs(pf, metric, "qrc", comp)
        if len(j) < 5:
            continue
        d = j["diff"].values
        try:
            _, p = stats.wilcoxon(d, alternative="two-sided")
        except Exception:
            p = np.nan
        out.append(dict(contrast=f"qrc-{comp}", n=len(d), mean_diff=float(d.mean()),
                        median_diff=float(np.median(d)),
                        p_wilcoxon=float(p) if np.isfinite(p) else np.nan))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Output/harness")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "auprc", "auroc_ewma"])
    ap.add_argument("--out", default="Output/harness")
    a = ap.parse_args()
    pf = _load_all(a.root)
    cells = pf.groupby(["dataset", "modality"]).ngroups
    print(f"loaded {len(pf)} per-fold rows across {cells} cells; "
          f"methods={sorted(pf.method.unique())}")

    tbl = master_table(pf, a.metric)
    print(f"\n=== MASTER subject-mean {a.metric} (per cell) ===")
    print(tbl.to_string())

    pct = per_cell_tost(pf, a.metric)
    if len(pct):
        print(f"\n=== per-cell TOST(QRC vs ESN) [{a.metric}] ===")
        print(pct[["dataset", "modality", "margin_pp", "mean_diff",
                   "ci90_lo", "ci90_hi", "p_tost", "equivalent"]].to_string(index=False))

    beq = benchmark_equivalence(pf, a.metric)
    print(f"\n=== BENCHMARK equivalence QRC vs ESN [{a.metric}] ===")
    print("(lead with SUBJ-AVG; POOLED is a max-power sensitivity check)")
    print(beq.to_string(index=False))

    bq = best_quantum(pf, a.metric)
    if len(bq):
        print(f"\n=== paired diffs QRC - X [{a.metric}] ===")
        print(bq.to_string(index=False))

    os.makedirs(a.out, exist_ok=True)
    tbl.to_csv(os.path.join(a.out, f"CONSOLIDATED_master_{a.metric}.csv"))
    if len(pct): pct.to_csv(os.path.join(a.out, f"CONSOLIDATED_percell_tost_{a.metric}.csv"), index=False)
    beq.to_csv(os.path.join(a.out, f"CONSOLIDATED_benchmark_equiv_{a.metric}.csv"), index=False)
    if len(bq): bq.to_csv(os.path.join(a.out, f"CONSOLIDATED_bestquantum_{a.metric}.csv"), index=False)
    print(f"\nwrote CONSOLIDATED_* to {a.out}")

if __name__ == "__main__":
    main()