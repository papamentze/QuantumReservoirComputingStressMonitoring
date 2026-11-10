"""
qrc_monitor_noisy.py -- WESAD benchmark at nominal noise (kappa = 1) for the recurrent and memoryless QRC.
Same folds, seeds, coupling draws, encoder, calibration and alarm logic as qrc_monitor_analysis.py;
only the noise flag differs. Writes to its own folder (V4_*.csv) so the benchmark results are untouched.

  python qrc_monitor_noisy.py run --jobs 8
  python qrc_monitor_noisy.py analyze
""" 
import os, glob, time, argparse, dataclasses
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import qrc_monitor_analysis as mon
from qrc_base import SuperNodeEncoder, conformal_pvalues, ranking_metrics
from qrc_core import QRCConfig
from qrc_harness import load_stream, _MODEL_FACTORY, _ewma

CELLS = ["wesad:hrv", "wesad:eda", "wesad:resp"]
NOISY = {"qrc_noisy": False, "qrc_nomem_noisy": True}            # True = leak set to 1
PAIRS = (("qrc_nomem_noisy", "qrc_nomem"),                        # memoryless: kappa 1 vs 0
         ("qrc_noisy", "qrc"),                                    # recurrent:  kappa 1 vs 0
         ("qrc_nomem_noisy", "qrc_noisy"))                        # recurrence on/off under noise


def fold(t, ix, method, seed, cfg, qcfg, X, feats, y, s, subj, lf):
    rng = np.random.default_rng([seed, ix])                       # identical split to qrc_monitor_analysis
    tr = np.array([u for u in subj if u != t]); rng.shuffle(tr)
    n_cal = max(1, int(round(cfg.cal_frac * len(tr)))); cal, fit = tr[:n_cal], tr[n_cal:]
    enc = SuperNodeEncoder(feats, cfg).fit(X[np.isin(s, fit)])

    def stream(u):
        m = s == u
        return enc.transform_angles(X[m], gamma=cfg.gamma), (y[m] == "STRESS").astype(int)

    q = dataclasses.replace(qcfg, reservoir_seed=mon.SEED0 + seed)   # same coupling draw as the benchmark
    if NOISY[method]:
        q = dataclasses.replace(q, **{lf: 1.0})
    model = _MODEL_FACTORY["qrc"](cfg, q, True, 4, seed, True).fit([stream(u) for u in fit])  # noisy=True, rich=True
    w = q.washout
    a_cal = np.concatenate([np.asarray(model.score_stream(th))[w:][yy[w:] == 0] for th, yy in map(stream, cal)])
    th, yy = stream(t); sc, yy = np.asarray(model.score_stream(th))[w:], yy[w:]
    if len(a_cal) == 0 or len(np.unique(yy)) < 2:
        return None
    p, pi = conformal_pvalues(a_cal, sc)
    r = ranking_metrics(pi, yy)
    return dict(method=method, seed=seed, subject=t, auroc=r["auroc"], auprc=r["auprc"],
                auroc_ewma=roc_auc_score(yy, _ewma(pi, 5.0)),
                p_seq=p.astype(np.float32), y_seq=yy.astype(np.int8))


def run(a):
    t_all = time.time()
    for cell in a.cells:
        ds, mod = cell.split(":")
        cfg, X, feats, y, s, subj = load_stream(ds, mod, 0, seed=42, road_threshold=0.6)
        qcfg = QRCConfig(base=cfg); lf = mon.leak_field(qcfg)
        d = os.path.join(a.out, ds, mod); os.makedirs(d, exist_ok=True)
        for method in NOISY:
            for seed in a.seeds:
                f = os.path.join(d, f"{method}_seed{seed}.pkl")
                if os.path.exists(f):
                    continue
                t0 = time.time()
                rows = Parallel(n_jobs=a.jobs)(delayed(fold)(t, ix, method, seed, cfg, qcfg, X, feats, y, s, subj, lf)
                                               for ix, t in enumerate(subj))
                df = pd.DataFrame([r for r in rows if r is not None]); df.to_pickle(f)
                print(f"{ds}/{mod} {method} seed {seed}: AUROC {df.auroc.mean():.3f} "
                      f"EWMA {df.auroc_ewma.mean():.3f} ({time.time() - t0:.0f} s)", flush=True)
    print(f"total {(time.time() - t_all) / 60:.1f} min", flush=True)


def analyze(a):
    fr = []
    for out, meths in ((a.out, tuple(NOISY)), (a.benchmark, ("qrc", "qrc_nomem"))):
        for me in meths:
            for f in glob.glob(os.path.join(out, "wesad", "*", f"{me}_seed*.pkl")):
                fr.append(pd.read_pickle(f).assign(modality=os.path.basename(os.path.dirname(f))))
    D = pd.concat(fr, ignore_index=True)
    i10 = int(np.flatnonzero(mon.ALPHAS == .10)[0])
    ext = []
    for p, yv in zip(D.p_seq, D.y_seq):
        p, yv = np.asarray(p, float), np.asarray(yv, int)
        c = mon.alarm_curve(p, yv, 3)[i10]
        ext.append((mon.lag1(p), c[0], c[1]))
    D[["score_lag1_autocorr", "far_pinned", "fa_per_h_pinned"]] = np.array(ext)
    MET = ["auroc", "auprc", "auroc_ewma", "score_lag1_autocorr", "far_pinned", "fa_per_h_pinned"]
    S = D.groupby(["modality", "method", "subject"])[MET].mean().reset_index()

    disc = S.groupby(["modality", "method"])[MET].mean().reset_index()
    disc.to_csv(os.path.join(a.out, "V4_discrimination.csv"), index=False)
    print("== V4 discrimination (qrc / qrc_nomem rows must equal Table 1) ==")
    print(disc.round(3).to_string(index=False))

    R = []
    for mt in MET:
        P = S.pivot_table(index=["modality", "subject"], columns="method", values=mt)
        for A, B in PAIRS:
            if A not in P or B not in P:
                continue
            d = (P[A] - P[B]).dropna()
            blk = [dict(metric=mt, contrast=f"{A}-{B}", unit=f"wesad-{mod}", n=len(g), diff=g.mean(),
                        p=mon.wilcoxon_p(g)) for mod, g in d.groupby(level=0)]
            for b, ph in zip(blk, mon.holm([b["p"] for b in blk])):
                b["p_holm"] = ph
            u = d.groupby(level=1).mean()
            r = dict(metric=mt, contrast=f"{A}-{B}", unit="wesad (subject-averaged)", **mon.pooled(u, mt))
            if mt in mon.AUC_METRICS and len(u) > 2 and u.std() > 0:
                se, k = u.std(ddof=1) / np.sqrt(len(u)), len(u) - 1
                r["p_tost_1pp"] = max(stats.t.sf((u.mean() + .01) / se, k), stats.t.cdf((u.mean() - .01) / se, k))
            R += blk + [r]
    con = pd.DataFrame(R)
    con.to_csv(os.path.join(a.out, "V4_contrasts.csv"), index=False)
    print("\n== V4 contrasts (AUROC and lag-1) ==")
    print(con[con.metric.isin(["auroc", "score_lag1_autocorr"])].round(4).to_string(index=False))

    # Ablation-style gap test on the full-readout model: |QRC - QRC_eps1| per seed x cell, kappa 0 vs 1
    G = D.groupby(["modality", "seed", "method"]).auroc.mean().unstack("method")
    g0 = (G["qrc"] - G["qrc_nomem"]).abs(); g1 = (G["qrc_noisy"] - G["qrc_nomem_noisy"]).abs()
    sh = (g0 - g1).dropna()
    print(f"\n== gap |QRC - QRC_eps1|: {g0.mean():.4f} (kappa=0) -> {g1.mean():.4f} (kappa=1); "
          f"shrinks in {(sh > 0).sum()} of {len(sh)} seed-cell pairs; Wilcoxon p = {mon.wilcoxon_p(sh):.3g}")
    inv = (G["qrc_nomem_noisy"] - G["qrc_nomem"]).abs()
    print(f"== memoryless |AUROC(kappa=1) - AUROC(kappa=0)|: max {inv.max():.5f}, mean {inv.mean():.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--out", default="Output/V2/monitor_noisy")
    ap.add_argument("--benchmark", default="Output/V2/monitor")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(mon.SEEDS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run(a) if a.mode == "run" else analyze(a) 