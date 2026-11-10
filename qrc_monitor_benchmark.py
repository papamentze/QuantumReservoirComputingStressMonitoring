"""
qrc_monitor_benchmark.py -- full-stream continuous-monitoring benchmark (run stage).

Methods: memoryless logistic regression, classical ESN and QRC (full readout + skip), five seeds.
Each seed changes the fit/calibration split, the ESN draw and the QRC coupling draw. LOSO by subject
(WESAD) / driver (AffectiveROAD); encoder fitted per fold on fit subjects only; full time-ordered
streams; the first 5 (washout) windows dropped for every method; conformal calibration on the
non-stress windows of held-out calibration subjects, scored inside their own full stream.
The paper's results are produced by qrc_monitor_analysis.py from the pickles written here
(this script's analyze mode writes earlier V2_* summaries that the paper does not use).
Writes ONLY to --out (resumable); refuses to write inside Output/FINAL.

  python qrc_monitor_benchmark.py run --out Output/V2/monitor --jobs 3
""" 
import os, sys, glob, time, argparse, dataclasses, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qrc_base import SuperNodeEncoder, conformal_pvalues, ranking_metrics, bootstrap_ci
from qrc_core import QRCConfig
from qrc_harness import load_stream, _MODEL_FACTORY, _ewma
warnings.filterwarnings("ignore")

CELLS = ["wesad:hrv", "wesad:eda", "wesad:resp", "road:hrv", "road:eda"]
METHODS, SEEDS, SEED0 = ("memoryless", "esn", "qrc"), (42, 3, 123, 37, 73), 20260408
ALPHAS, MS, HOP = np.round(np.arange(.02, .401, .02), 2), (1, 3, 5), 5.0     # alpha grid, debounce m, 5-s hop
FAR_BUDGETS, LAT_BUDGETS = (.05, .10, .15), (15., 25., 50.)                 # pre-specified; 15/25/50 s = 3/5/10 windows


def holm(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def wilcoxon_p(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan


def fold(t, ix, method, seed, cfg, qcfg, X, feats, y, s, subj):
    rng = np.random.default_rng([seed, ix])
    tr = np.array([u for u in subj if u != t]); rng.shuffle(tr)
    n_cal = max(1, int(round(cfg.cal_frac * len(tr)))); cal, fit = tr[:n_cal], tr[n_cal:]
    enc = SuperNodeEncoder(feats, cfg).fit(X[np.isin(s, fit)])

    def stream(u):
        m = s == u
        return enc.transform_angles(X[m], gamma=cfg.gamma), (y[m] == "STRESS").astype(int)

    q = dataclasses.replace(qcfg, reservoir_seed=SEED0 + seed)
    model = _MODEL_FACTORY[method](cfg, q, False, 4, seed, True).fit([stream(u) for u in fit])
    w = q.washout
    a_cal = np.concatenate([np.asarray(model.score_stream(th))[w:][yy[w:] == 0] for th, yy in map(stream, cal)])
    th, yy = stream(t); sc, yy = np.asarray(model.score_stream(th))[w:], yy[w:]
    if len(a_cal) == 0 or len(np.unique(yy)) < 2:
        return None
    p, pi = conformal_pvalues(a_cal, sc)
    r = ranking_metrics(pi, yy)
    return dict(method=method, seed=seed, subject=t, auroc=r["auroc"], auprc=r["auprc"],
                prevalence=r["prevalence"], auroc_ewma=roc_auc_score(yy, _ewma(pi, 5.0)),
                p_seq=p.astype(np.float32), y_seq=yy.astype(np.int8))


def run(a):
    for cell in a.cells:
        ds, mod = cell.split(":")
        cfg, X, feats, y, s, subj = load_stream(ds, mod, 0, seed=42, road_threshold=0.6)   # cap 0 = full streams
        qcfg = QRCConfig(base=cfg)
        d = os.path.join(a.out, ds, mod); os.makedirs(d, exist_ok=True)
        for method in METHODS:
            for seed in a.seeds:
                f = os.path.join(d, f"{method}_seed{seed}.pkl")
                if os.path.exists(f):
                    continue
                t0 = time.time()
                rows = Parallel(n_jobs=a.jobs)(delayed(fold)(t, ix, method, seed, cfg, qcfg, X, feats, y, s, subj)
                                               for ix, t in enumerate(subj))
                df = pd.DataFrame([r for r in rows if r is not None]); df.to_pickle(f)
                print(f"{ds}/{mod} {method} seed {seed}: AUROC {df.auroc.mean():.3f}  "
                      f"EWMA {df.auroc_ewma.mean():.3f}  ({time.time() - t0:.0f} s)", flush=True)


def alarm_curve(p, y, m):
    """Per alpha: FAR (fraction of non-stress windows in alarm), false alarms per hour,
    episode detection rate (stress episodes >= 3 windows), mean detection latency in seconds."""
    st = np.flatnonzero(np.diff(np.r_[0, y]) == 1); en = np.flatnonzero(np.diff(np.r_[y, 0]) == -1)
    keep = (en - st + 1) >= 3; st, en = st[keep], en[keep]
    hours = max((y == 0).sum() * HOP / 3600, 1e-9); res = np.full((len(ALPHAS), 4), np.nan)
    for i, al in enumerate(ALPHAS):
        alarm = np.convolve((p <= al).astype(int), np.ones(m, int))[:len(p)] >= m
        onset = alarm & ~np.r_[False, alarm[:-1]]
        lat = [np.argmax(alarm[a0:b0 + 1]) * HOP for a0, b0 in zip(st, en) if alarm[a0:b0 + 1].any()]
        res[i] = (alarm[y == 0].mean(), (onset & (y == 0)).sum() / hours,
                  len(lat) / len(st) if len(st) else np.nan, np.mean(lat) if lat else np.nan)
    return res


def lag1(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1] if np.std(x) > 0 else np.nan


def analyze(a):
    fr = []
    for f in glob.glob(os.path.join(a.out, "*", "*", "*.pkl")):
        ds, mod = f.replace("\\", "/").split("/")[-3:-1]
        fr.append(pd.read_pickle(f).assign(dataset=ds, modality=mod))
    D = pd.concat(fr, ignore_index=True)
    ck = ["dataset", "modality", "method", "subject"]
    S = D.groupby(ck)[["auroc", "auprc", "auroc_ewma", "prevalence"]].mean().reset_index()

    # Discrimination per cell and method: subject means with 95% bootstrap CI
    rng, T = np.random.default_rng(0), []
    for (ds, mod, me), g in S.groupby(["dataset", "modality", "method"]):
        r = dict(dataset=ds, modality=mod, method=me, n=len(g), prevalence=g.prevalence.mean())
        for mt in ("auroc", "auprc", "auroc_ewma"):
            r[mt] = g[mt].mean(); r[mt + "_lo"], r[mt + "_hi"] = bootstrap_ci(g[mt].values, 2000, .95, rng)
        T.append(r)
    pd.DataFrame(T).to_csv(os.path.join(a.out, "V2_discrimination.csv"), index=False)

    # QRC - X: per cell (Wilcoxon, Holm over cells) and subject-averaged over cells (90% CI, TOST at 5 pp)
    R = []
    for mt in ("auroc", "auprc", "auroc_ewma"):
        P = S.pivot_table(index=["dataset", "modality", "subject"], columns="method", values=mt)
        for comp in ("memoryless", "esn"):
            d = (P["qrc"] - P[comp]).dropna()
            block = [dict(metric=mt, contrast=f"qrc-{comp}", unit=f"{ds}-{mod}", n=len(g), diff=g.mean(),
                          p=wilcoxon_p(g)) for (ds, mod), g in d.groupby(level=[0, 1])]
            for b, ph in zip(block, holm([b["p"] for b in block])):
                b["p_holm"] = ph
            u = d.groupby(level=[0, 2]).mean(); n = len(u); se = u.std(ddof=1) / np.sqrt(n)
            hw = stats.t.ppf(.95, n - 1) * se
            p_tost = max(stats.t.sf((u.mean() + .05) / se, n - 1), stats.t.cdf((u.mean() - .05) / se, n - 1))
            block.append(dict(metric=mt, contrast=f"qrc-{comp}", unit="all (subject-averaged)", n=n, diff=u.mean(),
                              ci90_lo=u.mean() - hw, ci90_hi=u.mean() + hw, p=wilcoxon_p(u), p_tost_5pp=p_tost))
            R += block
    pd.DataFrame(R).to_csv(os.path.join(a.out, "V2_contrasts.csv"), index=False)

    # Alarm curves per subject, averaged over seeds
    C = {}
    for (ds, mod, me, su), g in D.groupby(ck):
        for m in MS:
            C[(ds, mod, me, su, m)] = np.nanmean([alarm_curve(np.asarray(p, float), np.asarray(yv, int), m)
                                                  for p, yv in zip(g.p_seq, g.y_seq)], axis=0)

    # Pinned operating point (alpha = 0.10, m = 3) + conformal validity + score smoothness
    i10, PIN = int(np.flatnonzero(ALPHAS == .10)[0]), []
    for (ds, mod, me), g in D.groupby(["dataset", "modality", "method"]):
        c = np.array([C[(ds, mod, me, su, 3)][i10] for su in g.subject.unique()])
        typ1 = [np.mean([(np.asarray(p)[np.asarray(yv) == 0] <= .10).mean() for p, yv in zip(h.p_seq, h.y_seq)])
                for _, h in g.groupby("subject")]
        ac1 = [np.nanmean([lag1(np.asarray(p, float)) for p in h.p_seq]) for _, h in g.groupby("subject")]
        PIN.append(dict(dataset=ds, modality=mod, method=me, far=np.nanmean(c[:, 0]),
                        false_alarms_per_h=np.nanmean(c[:, 1]), detection_rate=np.nanmean(c[:, 2]),
                        latency_s=np.nanmean(c[:, 3]), nonstress_windows_p_le_alpha=np.mean(typ1),
                        score_lag1_autocorr=np.nanmean(ac1)))
    pd.DataFrame(PIN).to_csv(os.path.join(a.out, "V2_alarm_pinned.csv"), index=False)

    # Matched comparisons: at equal FAR (detection rate, latency) and at equal latency (FAR), m = 1/3/5
    def pick(c, kind, tv):
        ok = (np.flatnonzero(c[:, 0] <= tv) if kind == "FAR"
              else np.flatnonzero((c[:, 2] >= .5) & (c[:, 3] <= tv)))
        return (c[ok[-1]] if kind == "FAR" else c[ok[0]]) if len(ok) else np.full(4, np.nan)

    MT = []
    for m in MS:
        fam = []
        for comp in ("esn", "memoryless"):
            for kind, budgets, col, name in (("FAR", FAR_BUDGETS, 2, "detection_rate"),
                                             ("FAR", FAR_BUDGETS, 3, "latency_s"),
                                             ("LAT", LAT_BUDGETS, 0, "far")):
                for tv in budgets:
                    per = {}
                    for (ds, mod, me, su, mm), cq in C.items():
                        if me != "qrc" or mm != m or (ds, mod, comp, su, m) not in C:
                            continue
                        va, vb = pick(cq, kind, tv)[col], pick(C[(ds, mod, comp, su, m)], kind, tv)[col]
                        if np.isfinite(va) and np.isfinite(vb):
                            per.setdefault((ds, su), []).append((va, vb))   # one unit per subject
                    arr = np.array([np.mean(v, axis=0) for v in per.values()]).reshape(-1, 2)
                    fam.append(dict(m=m, contrast=f"qrc-{comp}", matched=kind, budget=tv, outcome=name,
                                    n=len(arr), qrc=np.nanmean(arr[:, 0]), other=np.nanmean(arr[:, 1]),
                                    diff=np.nanmean(arr[:, 0] - arr[:, 1]), p=wilcoxon_p(arr[:, 0] - arr[:, 1])))
        for f, ph in zip(fam, holm([f["p"] for f in fam])):
            f["p_holm"] = ph
        MT += fam
    pd.DataFrame(MT).to_csv(os.path.join(a.out, "V2_alarm_matched.csv"), index=False)

    pd.set_option("display.width", 200)
    for name in ("V2_discrimination", "V2_contrasts", "V2_alarm_pinned"):
        print(f"\n== {name} =="); print(pd.read_csv(os.path.join(a.out, name + ".csv")).round(3).to_string(index=False))
    figure(C, a.out)


def figure(C, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.4), squeeze=False)
    for i, ds in enumerate(("wesad", "road")):
        for me, col, mk, lab in (("memoryless", "#56B4E9", "^", "memoryless"), ("esn", "#E69F00", "s", "classical ESN"),
                                 ("qrc", "#0072B2", "o", "QRC")):
            cs = np.array([v for k, v in C.items() if k[0] == ds and k[2] == me and k[4] == 3])
            if not len(cs):
                continue
            far = np.nanmean(cs[:, :, 0], 0)
            ax[i, 0].plot(far, np.nanmean(cs[:, :, 3], 0), "-" + mk, color=col, ms=2.5, label=lab)
            ax[i, 1].plot(far, np.nanmean(cs[:, :, 2], 0), "-" + mk, color=col, ms=2.5, label=lab)
        ax[i, 0].set_ylabel(f"{ds.upper()}: detection latency (s)"); ax[i, 1].set_ylabel("episode detection rate")
        for j in (0, 1):
            ax[i, j].set_xlabel("false-alarm rate (fraction of non-stress windows)")
    ax[0, 0].legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_operating_curve_v2.png"), dpi=600)
    fig.savefig(os.path.join(out, "fig_operating_curve_v2.tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--out", default="Output/V2/monitor")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()
    if os.path.abspath(a.out).startswith(os.path.abspath("Output/FINAL")):
        sys.exit("Refusing to write inside Output/FINAL.")
    os.makedirs(a.out, exist_ok=True)
    run(a) if a.mode == "run" else analyze(a) 