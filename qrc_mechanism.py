"""
qrc_mechanism.py -- recurrence x noise ablation of the quantum reservoir.
Does simulated NISQ noise change QRC monitoring AUROC through the reservoir's temporal memory,
and does the pattern hold across different random reservoirs?

Grid: 5 cells x D reservoir draws x leak eps in {0.15 = with memory, 1.0 = no recurrent state}
      x noise scale kappa in {0, 0.5, 1, 2, 3}.
Per configuration:
  * LOSO AUROC per held-out subject: base <Z>,<ZZ> reservoir (reduced readout), ridge readout fit on all
    other subjects, full time-ordered streams, washout 5 windows.
  * Linear memory capacity measured with a PERMUTED real drive (the modality's own angle distribution,
    i.i.d. in time; adjusted R^2), so MC reflects reservoir memory, not input autocorrelation.
    Built-in sanity check: eps = 1 must give MC ~ 0.
Design: native time order (no unstable sort), splits seeded without hash(), AffectiveROAD threshold 0.6.
Encoder: cohort-level and label-free, identical for every configuration (stated in the paper).
Writes ONLY to --out (resumable); refuses to write inside Output/FINAL.

  python qrc_mechanism.py run     --out Output/V2/mechanism --draws 5  --jobs 3
  python qrc_mechanism.py run     --out Output/V2/mechanism --draws 10 --jobs 3   # resumes, adds draws 5-9
  python qrc_mechanism.py analyze --out Output/V2/mechanism
"""
import os, sys, time, argparse, dataclasses, itertools, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qrc_base import Config, load_dataset, SuperNodeEncoder
from qrc_core import QRCConfig, QuantumReservoir, RidgeReadout
warnings.filterwarnings("ignore")

CELLS = ["wesad:hrv", "wesad:eda", "wesad:resp", "road:hrv", "road:eda"]
KAPPAS, LEAKS, SEED0 = (0.0, 0.5, 1.0, 2.0, 3.0), (0.15, 1.0), 20260408
KEY = ["dataset", "modality", "draw", "leak", "kappa"]


def holm(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def wilcoxon_p(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan


def scaled(cfg, k):
    return dataclasses.replace(cfg, depol_1q=cfg.depol_1q * k, depol_2q=cfg.depol_2q * k,
                               amp_damp=min(.999, cfg.amp_damp * k), phase_damp=min(.999, cfg.phase_damp * k),
                               readout_err=min(.5, cfg.readout_err * k))


def load_cell(cell):
    ds, mod = cell.split(":")
    cfg = Config(dataset=ds, modality=mod, n_qubits=6, seed=42, road_threshold=0.6).finalize()
    X, feats, y, s = load_dataset(cfg)
    keep = np.isin(y, ["BASE", "STRESS"]); X, y, s = X[keep], y[keep], s[keep]   # file order = time order
    enc = SuperNodeEncoder(feats, cfg).fit(X)
    th = enc.transform_angles(X, gamma=cfg.gamma)
    streams = {}
    for u in sorted(set(s)):
        yy = (y[s == u] == "STRESS").astype(int)
        if min(yy.sum(), (1 - yy).sum()) >= 3:
            streams[u] = (th[s == u], yy)
    return ds, mod, cfg, streams, len(enc.branches)


def reservoir_states(q, Q, k, th):
    return QuantumReservoir(q, noisy=k > 0, n_qubits=Q).run_sequence(th)


def loso_auroc(F, streams, w):
    out = {}
    for t in streams:
        tr = [u for u in streams if u != t]
        ro = RidgeReadout(1.0).fit(np.vstack([F[u][w:] for u in tr]),
                                   np.concatenate([streams[u][1][w:] for u in tr]))
        yt = streams[t][1][w:]
        if len(np.unique(yt)) == 2:
            out[t] = roc_auc_score(yt, ro.score(F[t][w:]))
    return out


def memory_capacity(q, Q, k, drive, w, max_delay=15):
    Xr = reservoir_states(q, Q, k, drive)[w:]
    u = drive.mean(1)[w:]; u = (u - u.mean()) / (u.std() + 1e-12)
    mc = 0.0
    for tau in range(1, max_delay + 1):
        A = np.hstack([Xr[tau:], np.ones((len(Xr) - tau, 1))]); b = u[:-tau]
        r2 = 1 - ((b - A @ np.linalg.lstsq(A, b, rcond=None)[0]) ** 2).sum() / ((b - b.mean()) ** 2).sum()
        n, p = len(b), A.shape[1] - 1
        mc += max(0.0, 1 - (1 - r2) * (n - 1) / (n - p - 1))      # adjusted R^2 removes in-sample bias
    return mc


def run(a):
    fa, fm = os.path.join(a.out, "mech_auroc.csv"), os.path.join(a.out, "mech_mc.csv")
    done = {tuple(map(str, r)) for r in pd.read_csv(fm)[KEY].values} if os.path.exists(fm) else set()
    for cell in a.cells:
        ds, mod, cfg, streams, Q = load_cell(cell)
        pool = np.vstack([v[0] for v in streams.values()])
        drives = [pool[np.random.default_rng(i).choice(len(pool), 400, replace=False)] for i in range(3)]
        for d, eps, k in itertools.product(range(a.draws), LEAKS, KAPPAS):
            if (ds, mod, str(d), str(eps), str(k)) in done:
                continue
            t0 = time.time()
            q = QRCConfig(base=scaled(cfg, k), leak=eps, reservoir_seed=SEED0 + d)
            F = dict(zip(streams, Parallel(n_jobs=a.jobs)(
                delayed(reservoir_states)(q, Q, k, v[0]) for v in streams.values())))
            au = loso_auroc(F, streams, q.washout)
            mc = float(np.mean(Parallel(n_jobs=a.jobs)(
                delayed(memory_capacity)(q, Q, k, dr, q.washout) for dr in drives)))
            pd.DataFrame([dict(dataset=ds, modality=mod, draw=d, leak=eps, kappa=k, subject=u, auroc=v)
                          for u, v in au.items()]).to_csv(fa, mode="a", header=not os.path.exists(fa), index=False)
            pd.DataFrame([dict(dataset=ds, modality=mod, draw=d, leak=eps, kappa=k, mc=mc)]).to_csv(
                fm, mode="a", header=not os.path.exists(fm), index=False)
            print(f"{ds}/{mod} draw {d} eps {eps} kappa {k}: AUROC {np.mean(list(au.values())):.3f}  "
                  f"MC {mc:.2f}  ({time.time() - t0:.0f} s)", flush=True)


def analyze(a):
    A = pd.read_csv(os.path.join(a.out, "mech_auroc.csv")).drop_duplicates(KEY + ["subject"], keep="last")
    M = pd.read_csv(os.path.join(a.out, "mech_mc.csv")).drop_duplicates(KEY, keep="last")
    g = ["dataset", "modality", "draw", "leak"]
    A = A.merge(A[A.kappa == 0][g + ["subject", "auroc"]].rename(columns={"auroc": "a0"}), on=g + ["subject"])
    A["d_auroc"] = A.auroc - A.a0
    M = M.merge(M[M.kappa == 0][g + ["mc"]].rename(columns={"mc": "mc0"}), on=g); M["d_mc"] = M.mc - M.mc0
    CD = A.groupby(KEY)[["auroc", "d_auroc"]].mean().reset_index().merge(M[KEY + ["mc", "d_mc"]], on=KEY)

    # Noise effect per cell, with and without memory, across reservoir draws
    rows = []
    for (ds, mod, eps, k), x in CD.groupby(["dataset", "modality", "leak", "kappa"]):
        subj = A[(A.dataset == ds) & (A.modality == mod) & (A.leak == eps) & (A.kappa == k)].groupby("subject").d_auroc.mean()
        rows.append(dict(dataset=ds, modality=mod, leak=eps, kappa=k, n_draws=len(x), auroc=x.auroc.mean(),
                         d_auroc=x.d_auroc.mean(), d_auroc_sd_draws=x.d_auroc.std(ddof=1),
                         share_draws_with_gain=(x.d_auroc > 0).mean(),
                         p_subjects=wilcoxon_p(subj) if k > 0 else np.nan,
                         mc=x.mc.mean(), mc_sd_draws=x.mc.std(ddof=1), d_mc=x.d_mc.mean()))
    S = pd.DataFrame(rows); S.to_csv(os.path.join(a.out, "mech_summary.csv"), index=False)

    # Memory mediation = noise effect WITH memory minus WITHOUT memory (per subject, draw-averaged)
    P = A[A.kappa > 0].groupby(["dataset", "modality", "kappa", "subject", "leak"]).d_auroc.mean().unstack("leak")
    med = pd.DataFrame([dict(dataset=ds, modality=mod, kappa=k, n=len(x),
                             memory_minus_nomemory=(x[0.15] - x[1.0]).mean(), p=wilcoxon_p(x[0.15] - x[1.0]))
                        for (ds, mod, k), x in P.groupby(level=[0, 1, 2])])
    med["p_holm"] = np.nan
    for _, idx in med.groupby("kappa").groups.items():
        med.loc[idx, "p_holm"] = holm(med.loc[idx, "p"])
    med.to_csv(os.path.join(a.out, "mech_mediation.csv"), index=False)

    # Does the AUROC / memory-capacity sign agreement hold reservoir by reservoir?
    x = CD[(CD.leak == 0.15) & (CD.kappa > 0)].copy()
    x["agree"] = np.sign(x.d_auroc) == np.sign(x.d_mc)
    sg = []
    for k, z in x.groupby("kappa"):
        per = z.groupby("draw").agree.agg(["sum", "size"])
        sg.append(dict(kappa=k, draws=len(per), mean_cells_agreeing=per["sum"].mean(),
                       draws_where_all_cells_agree=int((per["sum"] == per["size"]).sum()),
                       spearman_dAUROC_dMC=stats.spearmanr(z.d_auroc, z.d_mc).correlation))
    pd.DataFrame(sg).to_csv(os.path.join(a.out, "mech_sign_agreement.csv"), index=False)

    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False)); print(med.round(4).to_string(index=False))
    print(pd.DataFrame(sg).round(3).to_string(index=False))
    figure(S, a.out)


def figure(S, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})
    cells = list(dict.fromkeys(zip(S.dataset, S.modality)))
    fig, ax = plt.subplots(2, len(cells), figsize=(7.0, 3.2), sharex=True, squeeze=False)
    for j, (ds, mod) in enumerate(cells):
        for eps, ls, lab in ((0.15, "-", "with memory (ε = 0.15)"), (1.0, "--", "no memory (ε = 1)")):
            z = S[(S.dataset == ds) & (S.modality == mod) & (S.leak == eps)].sort_values("kappa")
            ax[0, j].plot(z.kappa, z.d_auroc, ls, color="#0072B2", marker="o", ms=2.5, label=lab)
            ax[0, j].fill_between(z.kappa, z.d_auroc - z.d_auroc_sd_draws, z.d_auroc + z.d_auroc_sd_draws,
                                  color="#0072B2", alpha=.15)
            ax[1, j].errorbar(z.kappa, z.mc, yerr=z.mc_sd_draws, fmt=ls, color="#D55E00", marker="o", ms=2.5,
                              capsize=1.5)
        ax[0, j].axhline(0, color="0.6", lw=.6); ax[0, j].set_title(f"{ds.upper()}-{mod.upper()}")
        ax[1, j].set_xlabel("noise scale κ")
    ax[0, 0].set_ylabel("ΔAUROC vs. κ = 0"); ax[1, 0].set_ylabel("memory capacity")
    ax[0, 0].legend(frameon=False, fontsize=6); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_noise_memory.png"), dpi=600)
    fig.savefig(os.path.join(out, "fig_noise_memory.tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--out", default="Output/V2/mechanism")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--cells", nargs="+", default=CELLS)
    a = ap.parse_args()
    if os.path.abspath(a.out).startswith(os.path.abspath("Output/FINAL")):
        sys.exit("Refusing to write inside Output/FINAL.")
    os.makedirs(a.out, exist_ok=True)
    run(a) if a.mode == "run" else analyze(a) 