"""
================================================================================
 qrc_experiment.py -- Experiment driver (QRC autonomic stress)
 NOTE: the paper's pipeline uses only ClassicalESN from this module. The experiments described
 below (E1-E5) belong to an earlier noise study and are not reported in the paper.

 Produces the paper's evidence:

  (E1) MAIN LOSO TABLE  -- clean vs NISQ QRC, per modality, subject-level
       bootstrap CIs (reused). Honest: AUPRC read against prevalence.

  (E2) NOISE-AS-RESOURCE CURVE  -- the headline. Sweep a single global noise
       scale kappa in [0, ..] that multiplies the channel strengths,
       and plot subject-mean AUROC / memory-proxy vs kappa. If a bump appears at
       kappa>0 (noise HELPS), that is the transfer result; if it is flat/monotone
       down, we report THAT honestly (still novel: first QRC characterization).

  (E3) MEMORY CAPACITY PROBE  -- a label-free reservoir diagnostic: linear
       short-term memory capacity (Jaeger) of the reservoir observables vs a
       random input, measured clean vs noisy. This explains WHY noise helps or
       not, independent of the stress labels -- the a-priori mechanism.

  (E4) BASELINES for context (not supremacy): a classical Echo State Network
       (ESN) reservoir of matched size, and a static (memoryless) ridge on the
       same features -- isolates the contribution of (a) quantumness and (b)
       temporal memory.

  (E5) TOST vs the static readout -- qrc_base.tost_paired to state, in the
       paper's own established language, whether the temporal reservoir is at
       least equivalent to (and where, better than) the memoryless model.

 All reservoir hyperparameters are PINNED (qrc_core.QRCConfig). The ONLY trained
 object is the linear readout, on FIT subjects only. Nothing is selected on the
 test metric or the noise level.

 USAGE
   python qrc_experiment.py --modality hrv --seeds 42 7 123 --jobs 3 --out Output/qrc_hrv
   python qrc_experiment.py --all-modalities --seeds 42 7 123 2 11 --jobs 3 --noise-sweep
   python qrc_experiment.py --smoke        # tiny synthetic end-to-end
================================================================================
"""

from __future__ import annotations
import os, sys, time, json, argparse, dataclasses
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
import hashlib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qrc_base as C
from qrc_base import (Config, load_dataset, bootstrap_ci, tost_paired,
                      SuperNodeEncoder, conformal_pvalues, ranking_metrics,
                      calibration_metrics, alarm_metrics)
import qrc_core as Q
from qrc_core import QRCConfig, QuantumReservoir, run_fold_qrc, RidgeReadout

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

METRICS = ["auroc", "auprc", "bacc", "ece", "nll", "brier", "far"]

# FAST PATH: cache each subject's reservoir features ONCE per noise level.

#  The reservoir web is drawn from a FIXED seed and (with a cohort-level, purely
#  UNSUPERVISED encoder) a subject's reservoir features depend only on
#  (noise level, subject) -- not on the LOSO seed, not on which subject is held
#  out. So we compute each subject's features once per distinct noise level, cache
#  them, and reuse across every seed AND across E1 + E2. The linear readout is
#  still fit strictly within each LOSO train split (no label leakage). Identical
#  reservoir maths; the only deviation from the strict original is the cohort-level
#  (vs per-fold) unsupervised encoder. Use strict=True to reproduce the original.
# ==============================================================================
def _noise_tag(cfg, noisy):
    if not noisy:
        return "clean"
    key = (f"{cfg.depol_1q:.6g}|{cfg.depol_2q:.6g}|{cfg.amp_damp:.6g}"
           f"|{cfg.phase_damp:.6g}|{cfg.readout_err:.6g}")
    return "noisy_" + hashlib.md5(key.encode()).hexdigest()[:8]

def _subject_feats_sm(reservoir, encoder, gamma, X, y_state, subjects, s, washout):
    """
    Full-block (BASE-first, for fit/test) + baseline-only (for cal) features for
    ONE subject. Stores BOTH the reservoir observables (fullX/baseX) AND the input
    angle features (fullF/baseF), aligned row-for-row, so the readout can be run in
    three modes from a single reservoir pass: features-only, reservoir-only, or
    reservoir+skip (concatenation).
    """
    m = (subjects == s); ys = y_state[m]
    keep = np.isin(ys, ["BASE", "STRESS"]); npn = int(keep.sum())
    nf = reservoir.n_features
    Qeff = len(encoder.branches)
    out = dict(fullX=np.empty((0, nf), np.float32), fully=np.empty(0, np.int8),
               fullF=np.empty((0, Qeff), np.float32),
               baseX=np.empty((0, nf), np.float32), baseF=np.empty((0, Qeff), np.float32),
               npn=np.int32(npn), nfeat=np.int32(nf))
    if npn:
        th = encoder.transform_angles(X[m][keep], gamma=gamma)
        order = np.argsort(~(ys[keep] == "BASE"))
        tho = th[order]
        yb = (ys[keep][order] == "STRESS").astype(int)
        Xr = reservoir.run_sequence(tho)[washout:]
        out["fullX"] = Xr.astype(np.float32); out["fully"] = yb[washout:].astype(np.int8)
        out["fullF"] = tho[washout:].astype(np.float32)          # skip-connection input
    keepb = (ys == "BASE")
    if keepb.sum():
        thb = encoder.transform_angles(X[m][keepb], gamma=gamma)
        Xb = reservoir.run_sequence(thb)[washout:]
        out["baseX"] = Xb.astype(np.float32)
        out["baseF"] = thb[washout:].astype(np.float32)
    return out

def _mode_matrix(d, block, mode):
    """
    Readout input matrix for a subject block ('full'/'base') under a mode:
    'feat' (input only), 'res'/'res+feat' (quantum reservoir), 'esn'/'esn+feat'
    (classical ESN reservoir). +feat modes concatenate the input (skip connection).
    """
    Xf = d[f"{block}F"].astype(float)
    if mode == "feat":
        return Xf
    if mode in ("res", "res+feat"):
        Xo = d[f"{block}X"].astype(float)
        return Xo if mode == "res" else (np.hstack([Xo, Xf]) if len(Xo) else Xo)
    Xe = d[f"{block}E"].astype(float)                       # classical ESN observables
    return Xe if mode == "esn" else (np.hstack([Xe, Xf]) if len(Xe) else Xe)

def _make_readout_sm(kind, ridge_lambda):
    if kind == "ridge":
        return RidgeReadout(ridge_lambda)
    mu = {}
    if kind == "logistic":
        class _LR:
            def fit(self, X, y):
                self.mu = X.mean(0); self.sg = X.std(0) + 1e-9
                self.c = LogisticRegression(max_iter=600).fit((X-self.mu)/self.sg, y); return self
            def score(self, X): return self.c.predict_proba((X-self.mu)/self.sg)[:, 1]
        return _LR()
    class _MLP:
        def fit(self, X, y):
            self.mu = X.mean(0); self.sg = X.std(0) + 1e-9
            self.c = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=500,
                                   random_state=0).fit((X-self.mu)/self.sg, y); return self
        def score(self, X): return self.c.predict_proba((X-self.mu)/self.sg)[:, 1]
    return _MLP()

def _precompute_sm(qcfg, X, feats, y_state, subjects, encoder, gamma, noisy,
                   cache_dir, jobs, resume=True):
    """
    Cache every subject's reservoir features for ONE noise level. Returns tag.
    Checkpoint = one npz per (noise-tag, subject); reused across seeds and E1/E2.
    """
    tag = _noise_tag(qcfg.base, noisy)
    Q_eff = len(encoder.branches)
    os.makedirs(cache_dir, exist_ok=True)
    subj = sorted(set(subjects))

    def _one(s):
        f = os.path.join(cache_dir, f"{tag}_subj_{s}.npz")
        if resume and os.path.exists(f):
            return
        res = QuantumReservoir(qcfg, noisy=noisy, n_qubits=Q_eff)
        fs = _subject_feats_sm(res, encoder, gamma, X, y_state, subjects, s, qcfg.washout)
        tmp = f + ".tmp.npz"; np.savez_compressed(tmp, **fs); os.replace(tmp, f)

    if jobs > 1 and _HAVE_JOBLIB:
        Parallel(n_jobs=jobs)(delayed(_one)(s) for s in subj)
    else:
        for s in subj:
            _one(s)
    return tag

def _loso_cached_sm(cfg, subjects, cache_dir, tag, cond, ridge_lambda):
    """
    Cheap LOSO from cached features for ONE (seed, noise-level, condition).
    Reproduces run_fold_qrc's split, ridge, conformal, and metric rows exactly.
    """
    subj = np.array(sorted(set(subjects)))
    cache = {s: dict(np.load(os.path.join(cache_dir, f"{tag}_subj_{s}.npz"))) for s in subj}
    rows = []
    for s_test in subj:
        rng = np.random.default_rng(cfg.seed + hash(s_test) % 10_000)
        train = np.array([u for u in subj if u != s_test]); rng.shuffle(train)
        n_cal = max(1, int(round(cfg.cal_frac * len(train))))
        cal_subj = set(train[:n_cal]); fit_subj = [u for u in train if u not in cal_subj]
        Xtr, ytr = [], []
        for s in fit_subj:
            d = cache[s]
            if int(d["npn"]) < 6 or not len(d["fullX"]):
                continue
            Xtr.append(d["fullX"].astype(float)); ytr.append(d["fully"])
        if not Xtr:
            continue
        readout = RidgeReadout(ridge_lambda).fit(np.vstack(Xtr), np.concatenate(ytr))
        a_cal = []
        for s in cal_subj:
            Xb = cache[s]["baseX"]
            if len(Xb):
                a_cal.append(readout.score(Xb.astype(float)))
        if not a_cal:
            continue
        a_cal = np.concatenate(a_cal)
        dt = cache[s_test]
        if int(dt["npn"]) < 6 or not len(dt["fullX"]):
            continue
        Xte = dt["fullX"].astype(float); is_stress = dt["fully"]
        if len(np.unique(is_stress)) < 2:
            continue
        a_test = readout.score(Xte)
        p, pi = conformal_pvalues(a_cal, a_test)
        rank = ranking_metrics(pi, is_stress)
        ece, nll, brier = calibration_metrics(pi, is_stress)
        latency, far = alarm_metrics(p, is_stress.astype(bool), cfg.alpha, cfg.debounce_m)
        rows.append(dict(arm=f"QRC:{cond}", subject=s_test, cond=cond,
                         auroc=rank["auroc"], auprc=rank["auprc"], bacc=rank["bacc"],
                         prevalence=rank["prevalence"], auprc_norm=rank["auprc_norm"],
                         ece=ece, nll=nll, brier=brier,
                         far=far, latency=(np.nan if latency is None else latency),
                         n_test=len(is_stress), n_features=int(dt["nfeat"])))
    return rows

def _loso_mode(cfg, subjects, cache, mode, ridge_lambda, readout_kind):
    """
    LOSO AUROC per test subject for ONE feature mode (direct AUROC on readout
    score -- the separability test, comparable to the E4 baselines).
    """
    subj = np.array(sorted(set(subjects)))
    per = {}
    for s_test in subj:
        rng = np.random.default_rng(cfg.seed + hash(s_test) % 10_000)
        train = np.array([u for u in subj if u != s_test]); rng.shuffle(train)
        Xtr, ytr = [], []
        for s in train:
            d = cache[s]
            if int(d["npn"]) < 6 or not len(d["fullX"]):
                continue
            Xtr.append(_mode_matrix(d, "full", mode)); ytr.append(d["fully"])
        dt = cache[s_test]
        if not Xtr or int(dt["npn"]) < 6 or not len(dt["fullX"]):
            continue
        yte = dt["fully"]
        if len(np.unique(yte)) < 2:
            continue
        ro = _make_readout_sm(readout_kind, ridge_lambda).fit(
            np.vstack(Xtr).astype(float), np.concatenate(ytr))
        sc = ro.score(_mode_matrix(dt, "full", mode).astype(float))
        try:
            per[s_test] = roc_auc_score(yte, sc)
        except Exception:
            pass
    return per

def run_ablation(qcfg, X, feats, y_state, subjects, seeds, jobs, cache_dir,
                 readout_kind="ridge"):
    """
    Does the untrained quantum reservoir add anything over the raw features?
    Runs LOSO in five modes on the CLEAN reservoir (from one cached quantum pass +
    a cheap classical ESN): features-only, quantum reservoir (-only / +skip), and
    classical ESN (-only / +skip). Adds per-subject paired significance for the two
    decisive tests. Returns a tidy comparison table.
    """
    encoder = SuperNodeEncoder(feats, qcfg.base).fit(X)
    gamma = qcfg.base.gamma
    tag = _precompute_sm(qcfg, X, feats, y_state, subjects, encoder, gamma,
                         noisy=False, cache_dir=cache_dir, jobs=jobs)
    subj = np.array(sorted(set(subjects)))
    cache = {s: dict(np.load(os.path.join(cache_dir, f"{tag}_subj_{s}.npz"))) for s in subj}
    # classical ESN reservoir (fixed draw, matched to the quantum reservoir's fixed
    # web) run on the SAME cached input features -> isolates quantum-vs-classical.
    esn = ClassicalESN(n_in=len(encoder.branches), leak=qcfg.leak,
                       seed=int(qcfg.reservoir_seed) % (2**32))
    for s in subj:
        d = cache[s]
        d["fullE"] = (esn.run_sequence(d["fullF"].astype(float)).astype(np.float32)
                      if len(d["fullF"]) else np.empty((0, esn.n_res), np.float32))
        d["baseE"] = (esn.run_sequence(d["baseF"].astype(float)).astype(np.float32)
                      if len(d["baseF"]) else np.empty((0, esn.n_res), np.float32))
    labels = {"feat": "FEATURES-ONLY (memoryless)",
              "res": "QUANTUM-RESERVOIR-ONLY", "res+feat": "QUANTUM-RESERVOIR+SKIP",
              "esn": "CLASSICAL-ESN-ONLY", "esn+feat": "CLASSICAL-ESN+SKIP"}
    per_mode = {}
    rows = []
    for mode in ["feat", "res", "res+feat", "esn", "esn+feat"]:
        per_seed = {}
        for sd in seeds:
            cfg_sd = dataclasses.replace(qcfg.base, seed=int(sd))
            for s, au in _loso_mode(cfg_sd, subjects, cache, mode, qcfg.ridge_lambda, readout_kind).items():
                per_seed.setdefault(s, []).append(au)
        vals = {s: float(np.mean(v)) for s, v in per_seed.items()}
        per_mode[mode] = vals
        arr = np.array(list(vals.values()))
        lo, hi = bootstrap_ci(arr, 1000, 0.95, np.random.default_rng(0)) if len(arr) else (np.nan, np.nan)
        rows.append(dict(arm=labels[mode], mode=mode, readout=readout_kind, n_subj=len(arr),
                         auroc_mean=round(float(arr.mean()), 3) if len(arr) else np.nan,
                         auroc_lo=round(float(lo), 3), auroc_hi=round(float(hi), 3)))
    df = pd.DataFrame(rows)
    try:
        def _paired(a, b):
            common = [s for s in per_mode[a] if s in per_mode[b]]
            da = np.array([per_mode[a][s] for s in common])
            db = np.array([per_mode[b][s] for s in common])
            dd = da - db
            p = float(wilcoxon(da, db).pvalue) if np.any(dd != 0) else np.nan
            return round(float(dd.mean()), 3), int((dd > 0).sum()), len(dd), (round(p, 4) if np.isfinite(p) else np.nan)
        for a, b, tag2 in [("res+feat", "feat", "QRC_skip_vs_features"),
                           ("res+feat", "esn+feat", "QRC_skip_vs_ESN_skip")]:
            md, npos, n, p = _paired(a, b)
            df.attrs[tag2] = dict(mean_delta=md, wins=npos, n=n, wilcoxon_p=p)
    except Exception:
        pass
    return df


def run_loso(qcfg: QRCConfig, X, feats, y_state, subjects, seeds, conditions, jobs=1,
             strict=False, cache_dir=None):
    """
    LOSO per seed -> tidy per-(seed,arm,subject) DataFrame.

    Default = FAST cached path (reservoir features computed once per noise level,
    reused across seeds and shared between E1 and E2). strict=True = original
    per-fold encoder + per-fold reservoir recompute (for exact reproduction).
    """
    subj = sorted(set(subjects))
    frames = []
    if strict:
        for sd in seeds:
            q_sd = dataclasses.replace(qcfg, base=dataclasses.replace(qcfg.base, seed=int(sd)))
            def _one(s):
                return run_fold_qrc(s, X, feats, y_state, subjects, q_sd, conditions)
            if jobs > 1 and _HAVE_JOBLIB:
                res = Parallel(n_jobs=jobs)(delayed(_one)(s) for s in subj)
            else:
                res = [_one(s) for s in subj]
            df = pd.DataFrame([r for fold in res for r in fold]); df["seed"] = int(sd)
            frames.append(df)
            print(f"   [seed {sd}] {len(df)} rows (strict)")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if cache_dir is None:
        cache_dir = os.path.join(".qrc_rescache", f"{qcfg.base.dataset}_{qcfg.base.modality}")
    encoder = SuperNodeEncoder(feats, qcfg.base).fit(X)      # cohort-level, seed-independent
    gamma = qcfg.base.gamma
    for cond in conditions:                                  # compute once per noise level
        _precompute_sm(qcfg, X, feats, y_state, subjects, encoder, gamma,
                       noisy=(cond == "noisy"), cache_dir=cache_dir, jobs=jobs)
    for sd in seeds:                                         # cheap per-seed LOSO on cache
        cfg_sd = dataclasses.replace(qcfg.base, seed=int(sd))
        seed_rows = []
        for cond in conditions:
            tag = _noise_tag(qcfg.base, cond == "noisy")
            seed_rows += _loso_cached_sm(cfg_sd, subjects, cache_dir, tag, cond, qcfg.ridge_lambda)
        df = pd.DataFrame(seed_rows); df["seed"] = int(sd); frames.append(df)
        print(f"   [seed {sd}] {len(df)} rows "
              f"(arms: {sorted(df.arm.unique()) if len(df) else '--'})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def seed_average(perfold):
    """
    Average each subject's metrics across seeds -> one row per (arm, subject).
    Variance reduction WITHOUT inflating n.
    """
    keep = [m for m in METRICS if m in perfold.columns] + \
           [c for c in ("prevalence", "auprc_norm", "latency") if c in perfold.columns]
    return perfold.groupby(["arm", "subject"], as_index=False)[keep].mean()

def summary_table(perfold_avg, n_boot=2000, level=0.95, seed=0):
    """Per-arm subject-level mean + bootstrap CI for each metric (reused CI)."""
    rng = np.random.default_rng(seed)
    out = []
    for arm, g in perfold_avg.groupby("arm"):
        row = dict(arm=arm, n_subj=g.subject.nunique())
        for m in METRICS:
            if m in g.columns:
                v = g[m].values
                lo, hi = bootstrap_ci(v, n_boot, level, rng)
                row[f"{m}_mean"] = float(np.nanmean(v))
                row[f"{m}_lo"] = lo; row[f"{m}_hi"] = hi
        out.append(row)
    return pd.DataFrame(out).sort_values("arm").reset_index(drop=True)

# E2: NOISE-AS-RESOURCE sweep  (the headline result)
def noise_sweep(qcfg: QRCConfig, X, feats, y_state, subjects, seeds,
                kappas=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0), jobs=1,
                strict=False, cache_dir=None):
    """
    Scale the channel strengths by kappa and record subject-mean AUROC
    / AUPRC at each level. kappa=0 is the clean reservoir; kappa=1 is EXACTLY the
    NISQ channel. This is the transfer test: does moderate noise raise
    the reservoir's separability (noise-as-resource) in this autonomic regime?

    Shares the reservoir-feature cache with E1 (kappa 0 and 1 are computed once).
    """
    base = qcfg.base
    rows = []
    for kappa in kappas:
        cfg_k = dataclasses.replace(
            base,
            depol_1q=base.depol_1q * kappa, depol_2q=base.depol_2q * kappa,
            amp_damp=min(0.999, base.amp_damp * kappa),
            phase_damp=min(0.999, base.phase_damp * kappa),
            readout_err=min(0.5, base.readout_err * kappa))
        q_k = dataclasses.replace(qcfg, base=cfg_k)
        cond = ("noisy",) if kappa > 0 else ("clean",)
        pf = run_loso(q_k, X, feats, y_state, subjects, seeds, cond, jobs=jobs,
                      strict=strict, cache_dir=cache_dir)
        if not len(pf):
            continue
        avg = seed_average(pf)
        for m in ("auroc", "auprc", "bacc"):
            v = avg[m].values
            rng = np.random.default_rng(0)
            lo, hi = bootstrap_ci(v, base.n_boot, base.ci_level, rng)
            rows.append(dict(kappa=kappa, metric=m, mean=float(np.nanmean(v)),
                             lo=lo, hi=hi, n_subj=avg.subject.nunique()))
        print(f"   [kappa={kappa:.2f}] AUROC={np.nanmean(avg['auroc']):.3f} "
              f"AUPRC={np.nanmean(avg['auprc']):.3f}")
    return pd.DataFrame(rows)

# E3: MEMORY CAPACITY probe  (label-free mechanism)
def memory_capacity(qcfg: QRCConfig, Q_eff, noisy, n_steps=400, max_delay=15, seed=0,
                    drive_thetas=None):
    """
    Short-term linear memory capacity of the reservoir observables.

    TWO MODES:
      * data-driven (drive_thetas given): the reservoir is driven with the REAL
        modality's time-ordered input-angle sequence, and for each delay tau we
        measure how well the *scalar summary of the past real input* u[t-tau] is
        linearly recoverable from the current reservoir state x[t]. This makes the
        memory curve genuinely MODALITY-SPECIFIC (fixes the bug where two
        modalities with equal Q_eff produced identical curves), and strengthens
        the mechanism: it is the recoverable memory of that modality's own dynamics.
      * generic (drive_thetas None): fallback i.i.d. probe (reservoir property at a
        given width); kept for smoke tests only.

    The scalar target u is the mean across the Q input angles at each step -- a
    label-free, modality-specific 1-D projection of the real drive. MC = sum_tau
    r^2(tau). Returns (MC_total, per-delay r2 array).
    """
    rng = np.random.default_rng(seed)
    if drive_thetas is not None:
        thetas = np.asarray(drive_thetas, dtype=float)
        if len(thetas) > n_steps:                     # cap length for cost, keep order
            start = rng.integers(0, len(thetas) - n_steps + 1)
            thetas = thetas[start:start + n_steps]
        u = thetas.mean(axis=1)                        # modality-specific scalar drive
        Qd = thetas.shape[1]
    else:
        u = rng.uniform(-1, 1, size=n_steps)
        thetas = np.pi * np.tanh(np.repeat(u[:, None], Q_eff, axis=1))
        Qd = Q_eff
    res = QuantumReservoir(qcfg, noisy=noisy, n_qubits=Qd)
    Xr = res.run_sequence(thetas)
    wash = qcfg.washout
    Xr = Xr[wash:]; u = u[wash:]
    # standardize target so r^2 is comparable across modalities with different scales
    u = (u - u.mean()) / (u.std() + 1e-9)
    r2s = []
    for tau in range(1, max_delay + 1):
        if tau >= len(u):
            break
        Xt = Xr[tau:]; target = u[:-tau]
        Xb = np.hstack([Xt, np.ones((len(Xt), 1))])
        w, *_ = np.linalg.lstsq(Xb, target, rcond=None)
        pred = Xb @ w
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum() + 1e-12
        r2s.append(max(0.0, 1 - ss_res / ss_tot))
    r2s = np.array(r2s)
    return float(r2s.sum()), r2s

def build_drive_sequence(X, feats, y_state, subjects, cfg, gamma=None, max_len=400):
    """
    Assemble a real, time-ordered input-angle drive for the memory probe from
    the ACTUAL modality data: concatenate subjects' BASE->STRESS windows in order,
    map through the SAME SuperNodeEncoder used in the task pipeline. This is what
    makes the memory curve modality-specific and mechanistically tied to the task.
    """
    enc = SuperNodeEncoder(feats, cfg).fit(X)
    g = cfg.gamma if gamma is None else gamma
    subj = sorted(set(subjects))
    seqs = []
    for s in subj:
        m = (subjects == s)
        ys = y_state[m]
        keep = np.isin(ys, ["BASE", "STRESS"])
        if keep.sum() < 4:
            continue
        order = np.argsort(~(ys[keep] == "BASE"))          # baseline first, then stress
        seqs.append(enc.transform_angles(X[m][keep][order], gamma=g))
        if sum(len(x) for x in seqs) >= max_len:
            break
    drive = np.vstack(seqs) if seqs else None
    return drive, len(enc.branches)

def memory_vs_noise(qcfg: QRCConfig, Q_eff, kappas=(0.0, 0.5, 1.0, 2.0, 3.0), seeds=(0, 1, 2),
                    drive_thetas=None):
    """
    Memory capacity vs noise. If drive_thetas (the real modality drive) is given,
    the curve is modality-specific (recommended); otherwise falls back to generic.
    """
    base = qcfg.base
    rows = []
    for kappa in kappas:
        cfg_k = dataclasses.replace(
            base, depol_1q=base.depol_1q * kappa, depol_2q=base.depol_2q * kappa,
            amp_damp=min(0.999, base.amp_damp * kappa),
            phase_damp=min(0.999, base.phase_damp * kappa),
            readout_err=min(0.5, base.readout_err * kappa))
        q_k = dataclasses.replace(qcfg, base=cfg_k)
        mcs = [memory_capacity(q_k, Q_eff, noisy=(kappa > 0), seed=s,
                               drive_thetas=drive_thetas)[0] for s in seeds]
        rows.append(dict(kappa=kappa, MC_mean=float(np.mean(mcs)), MC_std=float(np.std(mcs))))
        print(f"   [MC kappa={kappa:.2f}] memory capacity = {np.mean(mcs):.3f} +/- {np.std(mcs):.3f}")
    return pd.DataFrame(rows)

# E4: classical baselines (context, not supremacy)
class ClassicalESN:
    """
    Matched-size leaky Echo State Network reservoir (classical), same leak and
    similar spectral scale as the quantum reservoir, so the ONLY difference from
    QRC is quantum vs classical dynamics.
    """
    def __init__(self, n_in, n_res=64, leak=0.15, spectral=0.9, seed=0):
        rng = np.random.default_rng(seed)
        self.Win = rng.uniform(-1, 1, size=(n_res, n_in)) * 0.5
        W = rng.uniform(-1, 1, size=(n_res, n_res))
        radius = max(abs(np.linalg.eigvals(W)))
        self.W = W * (spectral / (radius + 1e-9))
        self.leak = leak; self.n_res = n_res

    def run_sequence(self, thetas):
        x = np.zeros(self.n_res)
        out = np.empty((len(thetas), self.n_res))
        for t, u in enumerate(thetas):
            pre = np.tanh(self.Win @ u + self.W @ x)
            x = (1 - self.leak) * x + self.leak * pre
            out[t] = x
        return out

# Orchestration
def run_all(modality, seeds, out_dir, jobs=1, do_noise=True, do_memory=True,
            synthetic=False, quick=False, n_qubits=6, dataset="wesad", strict=False,
            road_threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config(dataset=dataset, modality=modality, n_qubits=n_qubits,
                 synthetic=synthetic, quick=quick, seed=seeds[0],
                 road_threshold=road_threshold).finalize()
    X, feats, y_state, subjects = load_dataset(cfg)
    qcfg = QRCConfig(base=cfg)
    manifest = dict(dataset=dataset, modality=modality, seeds=list(seeds),
                    n_qubits=n_qubits,
                    encoder=("per_fold_strict" if strict else "cohort_unsupervised_cached"),
                    reservoir_feature_cache=(not strict),
                    reservoir=dataclasses.asdict(qcfg_public(qcfg)),
                    nisq=dict(depol_1q=cfg.depol_1q, depol_2q=cfg.depol_2q,
                              amp_damp=cfg.amp_damp, phase_damp=cfg.phase_damp,
                              readout_err=cfg.readout_err))

    print(f"\n=== E1 MAIN LOSO :: {modality} (clean+noisy) ===")
    t0 = time.time()
    rescache = os.path.join(out_dir, "_rescache")     # shared across E1 + E2 + seeds
    perfold = run_loso(qcfg, X, feats, y_state, subjects, seeds,
                       ("clean", "noisy"), jobs=jobs, strict=strict, cache_dir=rescache)
    perfold.to_csv(os.path.join(out_dir, f"perfold_{modality}.csv"), index=False)
    avg = seed_average(perfold)
    summ = summary_table(avg, n_boot=cfg.n_boot, level=cfg.ci_level)
    summ.to_csv(os.path.join(out_dir, f"summary_{modality}.csv"), index=False)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"   E1 done in {time.time()-t0:.0f}s")

    # E5 TOST clean-vs-noisy (equivalence, reused): does NISQ preserve QRC?
    tost = tost_paired(
        avg.rename(columns={"arm": "arm"}),
        "auroc",
        [("QRC:clean", "QRC:noisy")],
        cfg.tost_margins)
    if len(tost):
        tost.to_csv(os.path.join(out_dir, f"tost_clean_vs_noisy_{modality}.csv"), index=False)
        print("\n   TOST QRC:clean vs QRC:noisy (AUROC):")
        print(tost.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    sweep = mem = None
    Q_eff = None
    if do_noise:
        print(f"\n=== E2 NOISE-AS-RESOURCE SWEEP :: {modality} ===")
        sweep = noise_sweep(qcfg, X, feats, y_state, subjects, seeds, jobs=jobs,
                            strict=strict, cache_dir=rescache)
        sweep.to_csv(os.path.join(out_dir, f"noise_sweep_{modality}.csv"), index=False)
    if do_memory:
        print(f"\n=== E3 MEMORY CAPACITY vs NOISE :: {modality} (data-driven) ===")
        # DATA-DRIVEN drive: use the real modality sequence so the curve is
        # modality-specific (fixes the equal-Q_eff identical-curve artifact).
        drive, Q_eff = build_drive_sequence(X, feats, y_state, subjects, cfg)
        mem = memory_vs_noise(qcfg, Q_eff, drive_thetas=drive)
        mem.to_csv(os.path.join(out_dir, f"memory_capacity_{modality}.csv"), index=False)
        manifest["Q_eff"] = Q_eff
        manifest["memory_probe"] = "data_driven"

    with open(os.path.join(out_dir, f"manifest_{modality}.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return dict(perfold=perfold, summary=summ, sweep=sweep, memory=mem, Q_eff=Q_eff)

def qcfg_public(qcfg):
    """A copy without the heavy `base` for the manifest."""
    d = {k: v for k, v in dataclasses.asdict(qcfg).items() if k != "base"}
    @dataclasses.dataclass
    class _P: pass
    p = _P()
    for k, v in d.items():
        setattr(p, k, v)
    return p

# Which single modalities each benchmark actually provides. ROAD has NO
# respiration channel (HRV + EDA only); RESP is WESAD-only. Keeping this explicit
# means `--all-modalities` does the right thing per dataset and never tries to
# load a non-existent road_resp.csv.
MODALITIES_BY_DATASET = {
    "wesad": ["hrv", "eda", "resp"],
    "road": ["hrv", "eda"],
}

# The legacy dataset keeps the pre-namespace FLAT output layout (<out>/<mod>/) so an
# ALREADY-COMPLETED WESAD run is reused / updated in place -- no rerun forced, no
# parallel copy. New datasets (ROAD) write to <out>/<dataset>/<mod>/.
LEGACY_FLAT_DATASET = os.environ.get("QRC_LEGACY_FLAT_DATASET", "wesad")


def _resolve_out_dir(out_root, dataset, mod):
    """
    Where this (dataset, modality) writes. Legacy dataset -> flat <out>/<mod>/
    (matches an existing WESAD run); every other dataset -> <out>/<dataset>/<mod>/.
    """
    if dataset == LEGACY_FLAT_DATASET:
        return os.path.join(out_root, mod)
    return os.path.join(out_root, dataset, mod)


def main():
    ap = argparse.ArgumentParser(description="QRC experiment driver")
    ap.add_argument("--dataset", default="wesad", choices=["wesad", "road"],
                    help="benchmark; ROAD has HRV+EDA only (no RESP).")
    ap.add_argument("--modality", default="hrv", choices=["hrv", "eda", "resp"])
    ap.add_argument("--all-modalities", action="store_true",
                    help="run every modality the chosen --dataset provides.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--noise-sweep", action="store_true", help="run E2 (headline)")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--road-threshold", type=float, default=0.5, dest="road_threshold")
    ap.add_argument("--out", default="Output/qrc",
                    help="root; WESAD -> <out>/<modality>/ (legacy flat, reuses your "
                         "existing run); ROAD -> <out>/road/<modality>/.")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--strict-per-fold-encoder", action="store_true",
                    help="reproduce the ORIGINAL slow per-fold encoder + per-fold "
                         "reservoir recompute (no caching). Use to validate the fast path.")
    ap.add_argument("--ablation", action="store_true",
                    help="run the decisive test: features-only vs reservoir-only vs "
                         "reservoir+skip (does the quantum reservoir add value?).")
    ap.add_argument("--readout", default="ridge", choices=["ridge", "logistic", "mlp"],
                    help="readout for --ablation (reservoir stays untrained either way).")
    a = ap.parse_args()

    if a.smoke:
        a.synthetic = a.quick = True; a.seeds = [42, 7]; a.noise_sweep = True

    avail = MODALITIES_BY_DATASET[a.dataset]
    if a.all_modalities:
        mods = list(avail)
    else:
        if a.modality not in avail:
            raise SystemExit(
                f"[error] modality '{a.modality}' is not available for dataset "
                f"'{a.dataset}'. Available: {avail}.")
        mods = [a.modality]

    if a.ablation:
        allrows = []
        for mod in mods:
            out_dir = _resolve_out_dir(a.out, a.dataset, mod)
            cfg = Config(dataset=a.dataset, modality=mod, n_qubits=a.n_qubits,
                         synthetic=a.synthetic, quick=a.quick, seed=a.seeds[0],
                         road_threshold=a.road_threshold).finalize()
            X, feats, y_state, subjects = load_dataset(cfg)
            qcfg = QRCConfig(base=cfg)
            df = run_ablation(qcfg, X, feats, y_state, subjects, a.seeds, a.jobs,
                              os.path.join(out_dir, "_abl_cache"), readout_kind=a.readout)
            df.insert(0, "modality", mod); df.insert(0, "dataset", a.dataset)
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"ablation_{mod}_{a.readout}.csv"), index=False)
            allrows.append(df)
            print(f"\n=== ABLATION :: {a.dataset}/{mod}  (readout={a.readout}) ===")
            print(df[["arm", "auroc_mean", "auroc_lo", "auroc_hi", "n_subj"]].to_string(index=False))
            v = df.attrs.get("QRC_skip_vs_features", {})
            e = df.attrs.get("QRC_skip_vs_ESN_skip", {})
            if v:
                print(f"  QRC+skip vs features-only : Δ={v['mean_delta']:+.3f}, "
                      f"wins {v['wins']}/{v['n']}, Wilcoxon p={v['wilcoxon_p']}  "
                      f"({'reservoir helps' if v['mean_delta'] > 0.005 else 'no gain'})")
            if e:
                print(f"  QRC+skip vs classical-ESN+skip: Δ={e['mean_delta']:+.3f}, "
                      f"wins {e['wins']}/{e['n']}, Wilcoxon p={e['wilcoxon_p']}  "
                      f"({'QUANTUM beats classical' if e['mean_delta'] > 0.005 else 'no quantum advantage'})")
        pd.concat(allrows, ignore_index=True).to_csv(
            os.path.join(a.out, f"ablation_summary_{a.dataset}_{a.readout}.csv"), index=False)
        print("\nRead: reservoir+skip must EXCEED features-only for the quantum reservoir to earn its place.")
        return

    for mod in mods:
        out_dir = _resolve_out_dir(a.out, a.dataset, mod)
        print(f"[{a.dataset}/{mod}] -> {out_dir}")
        run_all(mod, a.seeds, out_dir, jobs=a.jobs,
                do_noise=a.noise_sweep or a.smoke, do_memory=not a.no_memory,
                synthetic=a.synthetic, quick=a.quick, n_qubits=a.n_qubits,
                dataset=a.dataset, strict=a.strict_per_fold_encoder)

    print("\n" + "#" * 80)
    print(" Reporting rules:")
    print("  * AUPRC is read against the fold prevalence (auprc_norm reported).")
    print("  * noise-as-resource is a TRANSFER test, not a discovery: report the")
    print("    kappa-curve as-is; if noise helps, contrast with other works where the")
    print("    SAME channel hurt; if it doesn't, that is still the first QRC")
    print("    characterization in this autonomic-stress regime.")
    print("  * reservoir is untrained -> no barren plateaus; only the linear")
    print("    readout is fit, on FIT subjects only.")

if __name__ == "__main__":
    main()