"""
qrc_monitor_analysis.py -- recurrence ablation on the monitoring benchmark and the full analysis.

run     : adds method "qrc_nomem": the benchmark QRC (same readout, coupling draw and fit/calibration
          split per seed) with the reservoir leak set to 1, i.e. no recurrence.
analyze : reads all benchmark pickles and writes the paper's results:
          V3_discrimination.csv (Table 1), V3_contrasts.csv (Table 2), V3_predictions.csv (Sect. 4.3),
          V3_alarm_pinned.csv (Table 3), V3_alarm_matched.csv and V3_alarm_null.csv (Sect. 4.4),
          fig_operating_curve_v3.* (Fig. 3).
Statistics: Wilcoxon signed-rank with Holm correction within pre-specified families, TOST at 5 pp,
circular-shift null with 20 rotations per stream.
Writes ONLY to --out; refuses to write inside Output/FINAL.

  python qrc_monitor_analysis.py run     --out Output/V2/monitor --methods qrc_nomem --jobs 8
  python qrc_monitor_analysis.py analyze --out Output/V2/monitor
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
BASE_METHODS, NEW_METHODS = ("memoryless", "esn", "qrc"), ("qrc_nomem",)
METHODS = BASE_METHODS + NEW_METHODS
SEEDS, SEED0 = (42, 3, 123, 37, 73), 20260408
ALPHAS, MS, HOP = np.round(np.arange(.02, .401, .02), 2), (1, 3, 5), 5.0
FAR_BUDGETS, LAT_BUDGETS = (.05, .10, .15), (15., 25., 50.)
CONTRASTS = (("qrc", "memoryless"), ("qrc", "esn"),                          # benchmark contrasts
             ("qrc_nomem", "qrc"), ("qrc_nomem", "memoryless"), ("qrc_nomem", "esn"))
OUTCOMES = (("FAR", FAR_BUDGETS, 2, "detection_rate", "primary"),          # primary outcomes
            ("FAR", FAR_BUDGETS, 3, "latency_s", "primary"),
            ("LAT", LAT_BUDGETS, 0, "far", "primary"),
            ("FAR", FAR_BUDGETS, 1, "false_alarms_per_h", "burden"))        # new, own Holm family
AUC_METRICS = ("auroc", "auprc", "auroc_ewma")
METRICS = AUC_METRICS + ("score_lag1_autocorr", "fa_per_h_pinned")
N_NULL = 20
LEAK_NAMES = ("leak", "leak_rate", "eps", "epsilon")


def holm(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def wilcoxon_p(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan


def leak_field(qcfg, override=None):
    names = [f.name for f in dataclasses.fields(qcfg)]
    hits = [override] if override else [n for n in names if n.lower() in LEAK_NAMES]
    if len(hits) != 1 or hits[0] not in names:
        sys.exit(f"Cannot identify the reservoir leak in QRCConfig; its fields are {names}. "
                 "Rerun with --leak-field <name>.")
    return hits[0]


def fold(t, ix, method, seed, cfg, qcfg, X, feats, y, s, subj, lf):
    rng = np.random.default_rng([seed, ix])                       # identical split to the benchmark
    tr = np.array([u for u in subj if u != t]); rng.shuffle(tr)
    n_cal = max(1, int(round(cfg.cal_frac * len(tr)))); cal, fit = tr[:n_cal], tr[n_cal:]
    enc = SuperNodeEncoder(feats, cfg).fit(X[np.isin(s, fit)])

    def stream(u):
        m = s == u
        return enc.transform_angles(X[m], gamma=cfg.gamma), (y[m] == "STRESS").astype(int)

    q = dataclasses.replace(qcfg, reservoir_seed=SEED0 + seed)    # same coupling draw as the benchmark "qrc"
    factory = method
    if method == "qrc_nomem":
        q, factory = dataclasses.replace(q, **{lf: 1.0}), "qrc"
    model = _MODEL_FACTORY[factory](cfg, q, False, 4, seed, True).fit([stream(u) for u in fit])
    w = q.washout                                                  # same windows dropped as the benchmark
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
        cfg, X, feats, y, s, subj = load_stream(ds, mod, 0, seed=42, road_threshold=0.6)
        qcfg = QRCConfig(base=cfg)
        lf = None
        if "qrc_nomem" in a.methods:
            lf = leak_field(qcfg, a.leak_field)
            print(f"   {ds}/{mod}: QRCConfig.{lf} = {getattr(qcfg, lf)} for 'qrc'; 'qrc_nomem' uses 1.0", flush=True)
        d = os.path.join(a.out, ds, mod); os.makedirs(d, exist_ok=True)
        for method in a.methods:
            for seed in a.seeds:
                f = os.path.join(d, f"{method}_seed{seed}.pkl")
                if os.path.exists(f):
                    continue
                t0 = time.time()
                rows = Parallel(n_jobs=a.jobs)(delayed(fold)(t, ix, method, seed, cfg, qcfg, X, feats, y, s, subj, lf)
                                               for ix, t in enumerate(subj))
                df = pd.DataFrame([r for r in rows if r is not None]); df.to_pickle(f)
                print(f"{ds}/{mod} {method} seed {seed}: AUROC {df.auroc.mean():.3f}  "
                      f"EWMA {df.auroc_ewma.mean():.3f}  ({time.time() - t0:.0f} s)", flush=True)


def alarm_curve(p, y, m):
    """Per alpha: FAR, false alarms per hour, episode detection rate (episodes >= 3 windows),
    mean detection latency in seconds."""
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


def null_curve(p, y, m, rng, n=N_NULL):
    """Circular-shift null: rotate the p-value stream by a random offset in [L/4, 3L/4]."""
    L = len(p)
    if L < 8:
        return np.full((len(ALPHAS), 4), np.nan)
    ks = rng.integers(L // 4, 3 * L // 4 + 1, size=n)
    return np.nanmean([alarm_curve(np.roll(p, k), y, m) for k in ks], axis=0)


def lag1(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1] if np.std(x) > 0 else np.nan


def pick(c, kind, tv):
    ok = (np.flatnonzero(c[:, 0] <= tv) if kind == "FAR"
          else np.flatnonzero((c[:, 2] >= .5) & (c[:, 3] <= tv)))
    return (c[ok[-1]] if kind == "FAR" else c[ok[0]]) if len(ok) else np.full(4, np.nan)


def pooled(u, metric):
    n = len(u)
    if n < 3:
        return dict(n=n, diff=u.mean() if n else np.nan)
    se = u.std(ddof=1) / np.sqrt(n); hw = stats.t.ppf(.95, n - 1) * se
    r = dict(n=n, diff=u.mean(), ci90_lo=u.mean() - hw, ci90_hi=u.mean() + hw, p=wilcoxon_p(u))
    if metric in AUC_METRICS:
        r["p_tost_5pp"] = max(stats.t.sf((u.mean() + .05) / se, n - 1), stats.t.cdf((u.mean() - .05) / se, n - 1))
    return r


def show(name, df):
    print(f"\n== {name} ==")
    print(df.round(3).to_string(index=False) if len(df) else "(nothing yet)")


def analyze(a):
    fr = []
    for f in glob.glob(os.path.join(a.out, "*", "*", "*.pkl")):
        ds, mod = f.replace("\\", "/").split("/")[-3:-1]
        fr.append(pd.read_pickle(f).assign(dataset=ds, modality=mod))
    D = pd.concat(fr, ignore_index=True)
    present = set(D.method.unique())
    print("methods found:", sorted(present), flush=True)
    ck = ["dataset", "modality", "method", "subject"]
    i10 = int(np.flatnonzero(ALPHAS == .10)[0])

    # Alarm curves per subject (mean over seeds) and their circular-shift null
    C, C0, rng0 = {}, {}, np.random.default_rng(7)
    for key, g in D.groupby(ck):
        st = [(np.asarray(p, float), np.asarray(yv, int)) for p, yv in zip(g.p_seq, g.y_seq)]
        for m in MS:
            C[key + (m,)] = np.nanmean([alarm_curve(p, yv, m) for p, yv in st], axis=0)
            C0[key + (m,)] = np.nanmean([null_curve(p, yv, m, rng0) for p, yv in st], axis=0)

    # Per-subject table (mean over seeds), plus score smoothness and pinned alarm burden
    S = D.groupby(ck)[["auroc", "auprc", "auroc_ewma", "prevalence"]].mean().reset_index()
    ac = pd.Series([lag1(np.asarray(p, float)) for p in D.p_seq], index=D.index, name="score_lag1_autocorr")
    S = S.merge(D[ck].join(ac).groupby(ck)["score_lag1_autocorr"].mean().reset_index(), on=ck)
    S["fa_per_h_pinned"] = [C[(r.dataset, r.modality, r.method, r.subject, 3)][i10, 1] for r in S.itertuples()]

    # Discrimination (benchmark rows reproduced exactly: same rng stream over the benchmark methods)
    T = []
    for meths, seed in ((BASE_METHODS, 0), (NEW_METHODS, 1)):
        rng = np.random.default_rng(seed)
        for (ds, mod, me), g in S[S.method.isin(meths)].groupby(["dataset", "modality", "method"]):
            r = dict(dataset=ds, modality=mod, method=me, n=len(g), prevalence=g.prevalence.mean())
            for mt in AUC_METRICS:
                r[mt] = g[mt].mean(); r[mt + "_lo"], r[mt + "_hi"] = bootstrap_ci(g[mt].values, 2000, .95, rng)
            T.append(r)
    disc = pd.DataFrame(T).sort_values(["dataset", "modality", "method"])
    disc.to_csv(os.path.join(a.out, "V3_discrimination.csv"), index=False)

    # Contrasts: per cell (Wilcoxon, Holm over cells), pooled over all subjects and over WESAD subjects
    R = []
    for mt in METRICS:
        P = S.pivot_table(index=["dataset", "modality", "subject"], columns="method", values=mt)
        for A, B in CONTRASTS:
            if A not in P or B not in P:
                continue
            d = (P[A] - P[B]).dropna()
            block = [dict(metric=mt, contrast=f"{A}-{B}", unit=f"{ds}-{mod}", n=len(g), diff=g.mean(),
                          p=wilcoxon_p(g)) for (ds, mod), g in d.groupby(level=[0, 1])]
            for b, ph in zip(block, holm([b["p"] for b in block])):
                b["p_holm"] = ph
            u = d.groupby(level=[0, 2]).mean()
            block.append(dict(metric=mt, contrast=f"{A}-{B}", unit="all (subject-averaged)", **pooled(u, mt)))
            if "wesad" in u.index.get_level_values(0):
                block.append(dict(metric=mt, contrast=f"{A}-{B}", unit="wesad (subject-averaged)",
                                  **pooled(u.xs("wesad", level=0), mt)))
            R += block
    Rdf = pd.DataFrame(R)
    Rdf.to_csv(os.path.join(a.out, "V3_contrasts.csv"), index=False)
    pred = Rdf[Rdf.contrast == "qrc_nomem-qrc"] if len(Rdf) else Rdf
    pred.to_csv(os.path.join(a.out, "V3_predictions.csv"), index=False)

    # Matched comparisons; family (scope=all, qrc, primary) is exactly the benchmark family
    MT = []
    for scope in ("all", "wesad"):
        for m in MS:
            fams = {}
            for A, B in CONTRASTS:
                if A not in present or B not in present:
                    continue
                cg = "qrc" if A == "qrc" else "qrc_nomem"
                for kind, budgets, col, name, og in OUTCOMES:
                    for tv in budgets:
                        per = {}
                        for k, ca in C.items():
                            ds, mod, me, su, mk = k
                            kb = (ds, mod, B, su, m)
                            if me != A or mk != m or kb not in C or (scope == "wesad" and ds != "wesad"):
                                continue
                            va, vb = pick(ca, kind, tv)[col], pick(C[kb], kind, tv)[col]
                            if np.isfinite(va) and np.isfinite(vb):
                                per.setdefault((ds, su), []).append((va, vb))      # one unit per subject
                        arr = np.array([np.mean(v, axis=0) for v in per.values()]).reshape(-1, 2)
                        fams.setdefault((cg, og), []).append(dict(
                            scope=scope, m=m, family=f"{cg}/{og}", contrast=f"{A}-{B}", matched=kind,
                            budget=tv, outcome=name, n=len(arr), a=np.nanmean(arr[:, 0]),
                            b=np.nanmean(arr[:, 1]), diff=np.nanmean(arr[:, 0] - arr[:, 1]),
                            p=wilcoxon_p(arr[:, 0] - arr[:, 1])))
            for fam in fams.values():
                for f, ph in zip(fam, holm([f["p"] for f in fam])):
                    f["p_holm"] = ph
                MT += fam
    pd.DataFrame(MT).to_csv(os.path.join(a.out, "V3_alarm_matched.csv"), index=False)

    # Circular-shift null: detection above each method's own null at matched FAR, and contrasts of that excess
    def excess(me, m, tv, scope):
        per = {}
        for k, c in C.items():
            ds, mod, mm, su, mk = k
            if mm != me or mk != m or (scope == "wesad" and ds != "wesad"):
                continue
            r_, z_ = pick(c, "FAR", tv)[2], pick(C0[k], "FAR", tv)[2]
            if np.isfinite(r_) and np.isfinite(z_):
                per.setdefault((ds, su), []).append((r_, z_))
        return {u: np.mean(v, axis=0) for u, v in per.items()}

    NL = []
    for scope in ("all", "wesad"):
        for m in MS:
            E, rows = {}, []
            for me in [x for x in METHODS if x in present]:
                for tv in FAR_BUDGETS:
                    E[(me, tv)] = ex = excess(me, m, tv, scope)
                    v = np.array(list(ex.values())).reshape(-1, 2)
                    rows.append(dict(scope=scope, m=m, row="method", name=me, budget=tv, n=len(v),
                                     detection=np.nanmean(v[:, 0]), detection_null=np.nanmean(v[:, 1]),
                                     excess=np.nanmean(v[:, 0] - v[:, 1]), p=wilcoxon_p(v[:, 0] - v[:, 1])))
            for r_, ph in zip(rows, holm([r_["p"] for r_ in rows])):
                r_["p_holm"] = ph
            fam = []
            for A, B in CONTRASTS:
                if A not in present or B not in present:
                    continue
                for tv in FAR_BUDGETS:
                    ea, eb = E[(A, tv)], E[(B, tv)]
                    ks = sorted(set(ea) & set(eb))
                    d = np.array([(ea[k][0] - ea[k][1]) - (eb[k][0] - eb[k][1]) for k in ks])
                    fam.append(dict(scope=scope, m=m, row="contrast", name=f"{A}-{B}", budget=tv, n=len(d),
                                    excess=np.mean(d) if len(d) else np.nan, p=wilcoxon_p(d)))
            for f, ph in zip(fam, holm([f["p"] for f in fam])):
                f["p_holm"] = ph
            NL += rows + fam
    null_df = pd.DataFrame(NL)
    null_df.to_csv(os.path.join(a.out, "V3_alarm_null.csv"), index=False)

    # Pinned operating point (alpha = 0.10, m = 3), including qrc_nomem
    PIN = []
    for (ds, mod, me), g in D.groupby(["dataset", "modality", "method"]):
        c = np.array([C[(ds, mod, me, su, 3)][i10] for su in g.subject.unique()])
        typ1 = [np.mean([(np.asarray(p)[np.asarray(yv) == 0] <= .10).mean() for p, yv in zip(h.p_seq, h.y_seq)])
                for _, h in g.groupby("subject")]
        ac1 = [np.nanmean([lag1(np.asarray(p, float)) for p in h.p_seq]) for _, h in g.groupby("subject")]
        PIN.append(dict(dataset=ds, modality=mod, method=me, far=np.nanmean(c[:, 0]),
                        false_alarms_per_h=np.nanmean(c[:, 1]), detection_rate=np.nanmean(c[:, 2]),
                        latency_s=np.nanmean(c[:, 3]), nonstress_windows_p_le_alpha=np.mean(typ1),
                        score_lag1_autocorr=np.nanmean(ac1)))
    pin = pd.DataFrame(PIN)
    pin.to_csv(os.path.join(a.out, "V3_alarm_pinned.csv"), index=False)

    pd.set_option("display.width", 220)
    show("V3_discrimination", disc)
    show("V3_predictions (qrc_nomem - qrc)", pred)
    show("V3_alarm_pinned", pin)
    show("V3_alarm_null (m = 3)", null_df[null_df.m == 3])
    figure(C, C0, a.out)

def figure(C, C0, out, m=3):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    sty = (("memoryless", "#56B4E9", "^", "logistic"), ("esn", "#E69F00", "s", "ESN"),
           ("qrc", "#0072B2", "o", "QRC (ε = 0.15)"), ("qrc_nomem", "#D55E00", "D", "QRC (ε = 1)"))
    rows = (("wesad", "WESAD (15 subjects)"), ("road", "AffectiveROAD (9 drivers): negative control"))
    fig = plt.figure(figsize=(4.8, 3.4), layout="constrained")
    handles = []
    for sf, (ds, title) in zip(fig.subfigures(2, 1), rows):
        ax = sf.subplots(1, 3)
        sf.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=7)
        for me, col, mk, lab in sty:
            cs = np.array([v for k, v in C.items() if k[0] == ds and k[2] == me and k[4] == m])
            if not len(cs):
                continue
            if lab not in [h.get_label() for h in handles]:
                handles.append(Line2D([], [], color=col, marker=mk, ms=2.5, lw=0.9, label=lab))
            far = np.nanmean(cs[:, :, 0], 0)
            for j, ci in ((0, 3), (1, 2), (2, 1)):
                ax[j].plot(far, np.nanmean(cs[:, :, ci], 0), "-" + mk, color=col, ms=1.8, lw=0.9)
            c0 = np.array([v for k, v in C0.items() if k[0] == ds and k[2] == me and k[4] == m])
            ax[1].plot(np.nanmean(c0[:, :, 0], 0), np.nanmean(c0[:, :, 2], 0), ":", color=col, lw=1.0)
        for j, lab in enumerate(("latency (s)", "detection rate", "false alarms / h")):
            ax[j].set_ylabel(lab); ax[j].set_xlim(0, .36)
            if ds == "road":
                ax[j].set_xlabel("FAR")
    handles.append(Line2D([], [], color="0.3", ls=":", lw=1.0, label="null"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    for ext, kw in ((".png", dict(dpi=600)), (".pdf", {}),
                    (".tiff", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"}))):
        fig.savefig(os.path.join(out, "fig_operating_curve_v3" + ext), **kw)
    plt.close(fig) 

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--out", default="Output/V2/monitor")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--leak-field", default=None, help="name of the leak field in QRCConfig (auto-detected)")
    a = ap.parse_args()
    if os.path.abspath(a.out).startswith(os.path.abspath("Output/FINAL")):
        sys.exit("Refusing to write inside Output/FINAL.")
    os.makedirs(a.out, exist_ok=True)
    run(a) if a.mode == "run" else analyze(a) 