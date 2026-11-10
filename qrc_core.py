"""
================================================================================
 qrc_core.py -- Quantum Reservoir Computing for subject-independent autonomic
                stress monitoring on wearable data  
                
 WHAT THIS IS (novelty, stated so it cannot drift into over-claiming)
   The first application of quantum reservoir computing (QRC) to autonomic
   wearable stress detection. A FIXED (untrained) quantum reservoir ingests a
   subject's windows AS A TIME-ORDERED SEQUENCE, one window per reservoir step,
   and its fading memory couples consecutive windows. A trained LINEAR readout
   maps reservoir observables to a stress score. Because the reservoir is not
   trained, there are NO variational parameters and therefore NO barren plateaus.

 HONEST SCOPE (enforced by the data we actually have)
   A reservoir step = one feature window; the temporal claim
   is at the BETWEEN-window scale (session dynamics), never sub-window. Windows
   are consecutive per (subject, condition), which is a valid monitoring stream.

 * From qrc_base: Config (incl. the EXACT NISQ channel constants), load_dataset,
   SuperNodeEncoder (fit-on-fit-only angle mapping), split-conformal p-values,
   alarm_metrics, calibration_metrics, ranking_metrics, bootstrap_ci, tost_paired.

 DESIGN (all reservoir hyperparameters PINNED a-priori; never tuned on the label)
   Encoding      : window -> Q autonomic super-node angles (SuperNodeEncoder),
                   theta_t = pi*tanh(gamma * g_t)
   Reservoir step: rho_t = N[ U(theta_t) . ( (1-eps) rho_{t-1} + eps rho_0 ) . U(theta_t)^dag ]
                   where U injects the input via RX/RZ + a FIXED random IsingZZ
                   coupling web (the untrained reservoir), N is the nominal noise
                   channel (clean run: N = identity), and the LEAK term
                   (1-eps)rho_{t-1}+eps rho_0 gives tunable-but-pinned fading memory.
   Readout feats : x_t = < Z_q >, < Z_a Z_b >_{coupled pairs} on rho_t  (real,
                   bounded, NISQ-robust). Linear/ridge readout, trained on FIT
                   subjects only. Anomaly score = readout stress-probability.
   Scoring path  : conformal p-values calibrated on BASELINE reservoir scores of
                   held-out calibration subjects -> alarm rule (alpha, m) reused.

================================================================================
"""

from __future__ import annotations
import os, sys, time, argparse, dataclasses
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # keep BLAS single-threaded under fold parallelism
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pennylane as qml
import qrc_base as C
from qrc_base import (Config, load_dataset, SuperNodeEncoder,
                      conformal_pvalues, alarm_metrics, calibration_metrics,
                      ranking_metrics, _coupling_edges)

# Reservoir Config (reservoir-only knobs)
@dataclasses.dataclass
class QRCConfig:
    """
    Reservoir hyperparameters. ALL pinned a-priori."""
    base: Config = None
    # reservoir dynamics (pinned)
    leak: float = 0.15            # eps: input-refresh / leak rate (fading memory strength)
    coupling_scale: float = 0.6   # scale of the FIXED random IsingZZ reservoir web
    input_scale: float = 1.0      # scale on the injected RX/RZ input rotations
    steps_per_window: int = 1     # reservoir sub-steps per input window (>=1; pinned 1)
    washout: int = 5              # initial reservoir steps discarded per session (transient)
    readout_second_order: bool = True   # include <Z_a Z_b> on coupled pairs in the feature vec
    reservoir_seed: int = 20260408      # SEPARATE, fixed seed for the random reservoir web
    # readout
    ridge_lambda: float = 1.0     # ridge penalty for the linear readout (pinned; not tuned on AUROC)

    def qubits(self):
        return self.base.n_qubits

# The Quantum Reservoir (fixed / untrained)
class QuantumReservoir:
    """
    A fixed quantum reservoir on `default.mixed` (density-matrix simulation).

    One reservoir step consumes one input angle vector theta (length Q) and
    updates the reservoir density matrix rho:

        rho <- (1-eps) rho + eps rho0                      # leak / input refresh
        rho <- U_res(theta) rho U_res(theta)^dag           # input-driven unitary
        rho <- NISQ_channel(rho)                           # identity if clean

    U_res encodes the input on every qubit (RX,RZ) and applies a FIXED random
    IsingZZ coupling web whose angles are drawn ONCE from `reservoir_seed` and
    never trained -- this is what makes it a reservoir (rich fixed dynamics),
    and what removes barren plateaus (no variational parameters).

    Readout observables x = [<Z_q>] (+ optional [<Z_a Z_b>] on coupled pairs).
    """

    def __init__(self, qcfg: QRCConfig, noisy=False, n_qubits=None):
        self.q = qcfg
        self.cfg = qcfg.base
        # Bind Q to the ACTUAL angle width the encoder produces (== number of
        # autonomic branches present, which can be < requested n_qubits). This
        # keeps input injection and reservoir wires exactly aligned with theta.
        self.Q = int(n_qubits) if n_qubits is not None else qcfg.qubits()
        self.noisy = noisy
        self.edges = _coupling_edges(self.Q, getattr(self.cfg, "entangle_topology", "ring"))
        # draw the FIXED random reservoir couplings once (untrained)
        rng = np.random.default_rng(qcfg.reservoir_seed)
        self.coupling = {e: float(qcfg.coupling_scale * rng.uniform(-np.pi, np.pi))
                         for e in self.edges}
        # a fixed random per-qubit bias makes the reservoir inhomogeneous (richer memory)
        self.bias = qcfg.input_scale * rng.uniform(-0.5, 0.5, size=self.Q)
        self._build()

    # circuit
    def _apply_input_and_reservoir(self, theta):
        s = self.q.input_scale
        for q in range(self.Q):
            qml.RX(s * theta[q] + self.bias[q], wires=q)          # input injection
            qml.RZ(0.5 * s * theta[q], wires=q)
        for (a, b) in self.edges:                                  # FIXED reservoir web
            qml.IsingZZ(self.coupling[(a, b)], wires=[a, b])
        if self.noisy:
            for q in range(self.Q):                                # nominal noise channel
                qml.DepolarizingChannel(self.cfg.depol_1q, wires=q)
                qml.AmplitudeDamping(self.cfg.amp_damp, wires=q)
                qml.PhaseDamping(self.cfg.phase_damp, wires=q)
            for (a, b) in self.edges:
                qml.DepolarizingChannel(self.cfg.depol_2q, wires=a)
                qml.DepolarizingChannel(self.cfg.depol_2q, wires=b)

    def _build(self):
        dev = qml.device("default.mixed", wires=self.Q)
        obs = [qml.PauliZ(q) for q in range(self.Q)]
        if self.q.readout_second_order:
            obs += [qml.PauliZ(a) @ qml.PauliZ(b) for (a, b) in self.edges]
        self._obs = obs

        @qml.qnode(dev, interface="numpy")
        def _step(rho_in, theta):
            qml.QubitDensityMatrix(rho_in, wires=range(self.Q))
            self._apply_input_and_reservoir(theta)
            return [qml.expval(o) for o in obs] + [qml.density_matrix(wires=range(self.Q))]
        self._step = _step
        self.n_features = len(obs)
        # ground state rho0 = |0..0><0..0|
        self.rho0 = np.zeros((2 ** self.Q, 2 ** self.Q), dtype=complex)
        self.rho0[0, 0] = 1.0

    # drive the reservoir over a session
    def run_sequence(self, thetas):
        """
        thetas: (T, Q) time-ordered input angles for ONE session/subject-block.
        Returns X_res: (T, n_features) reservoir observables per step (after
        washout the caller decides what to keep). Uses a leaky input-refresh.
        """
        eps = self.q.leak
        rho = self.rho0.copy()
        feats = np.empty((len(thetas), self.n_features), dtype=np.float64)
        for t, th in enumerate(thetas):
            rho = (1.0 - eps) * rho + eps * self.rho0          # leak toward ground state
            for _ in range(self.q.steps_per_window):
                out = self._step(rho, th)
                rho = np.asarray(out[-1])
            feats[t] = np.real(np.asarray(out[:-1], dtype=np.float64))
        return feats

# Ridge Readout  (trained on FIT subjects only; the ONLY trained component)
class RidgeReadout:
    """
    Closed-form ridge classifier on reservoir features -> stress probability.
    Trained on fit-subject reservoir states only. Platt-free: we map the ridge
    score through a logistic squashing calibrated on the fit set, then let the
    downstream split-conformal layer provide valid p-values. 
    Standardization stats are fit on the fit set only (no leakage)."""

    def __init__(self, lam=1.0):
        self.lam = float(lam)

    def fit(self, X, y):
        self.mu_ = X.mean(0); self.sd_ = X.std(0); self.sd_[self.sd_ == 0] = 1.0
        Xs = (X - self.mu_) / self.sd_
        Xb = np.hstack([Xs, np.ones((len(Xs), 1))])            # bias column
        d = Xb.shape[1]
        R = self.lam * np.eye(d); R[-1, -1] = 0.0              # don't penalize bias
        yc = (2.0 * y - 1.0)                                    # {0,1} -> {-1,+1}
        self.w_ = np.linalg.solve(Xb.T @ Xb + R, Xb.T @ yc)
        raw = Xb @ self.w_
        # logistic scale from fit residual spread (fixed, label-free-ish calibration)
        self.scale_ = 1.0 / (raw.std() + 1e-8)
        # orient so higher score => stress, using FIT labels only (no leakage)
        self.sign_ = 1.0
        if raw[y == 1].mean() < raw[y == 0].mean():
            self.sign_ = -1.0
        return self

    def score(self, X):
        Xs = (X - self.mu_) / self.sd_
        Xb = np.hstack([Xs, np.ones((len(Xs), 1))])
        raw = self.sign_ * (Xb @ self.w_)
        return 1.0 / (1.0 + np.exp(-self.scale_ * raw))        # stress probability in (0,1)

# Session Assembly  (build per-subject time-ordered monitoring streams)
def build_sessions(X, y_state, subjects, encoder, gamma, qcfg, subj_list):
    """
    For each subject in subj_list, build a time-ordered monitoring session:
    baseline windows first, then stress windows (a canonical rest->stress
    monitoring stream). Returns dict subj -> (thetas (T,Q), is_stress (T,),
    y01 (T,)). AMUSE is dropped for the binary stress task (kept out entirely).
    Order within a block is preserved as it appears in the CSV (consecutive
    windows), which is the monitoring order.
    """
    out = {}
    for s in subj_list:
        m = (subjects == s)
        Xs, ys = X[m], y_state[m]
        base_idx = np.where(ys == "BASE")[0]
        strs_idx = np.where(ys == "STRESS")[0]
        if len(base_idx) < 3 or len(strs_idx) < 3:
            continue
        order = np.concatenate([base_idx, strs_idx])           # rest -> stress stream
        thetas = encoder.transform_angles(Xs[order], gamma=gamma)
        is_stress = np.array([1] * len(base_idx) + [0] * len(strs_idx))  # placeholder
        is_stress = np.concatenate([np.zeros(len(base_idx), int),
                                    np.ones(len(strs_idx), int)])
        out[s] = dict(thetas=thetas, is_stress=is_stress, y01=is_stress.copy())
    return out

# One LOSO Fold
def run_fold_qrc(s_test, X, feats, y_state, subjects, qcfg: QRCConfig,
                 conditions=("clean", "noisy")):
    """
    Leave-subject-out fold. Fit encoder + reservoir readout on FIT subjects,
    calibrate conformal on held-out CAL subjects (baseline only), evaluate the
    monitoring stream on the TEST subject. Returns a list of per-arm dict rows
    (auroc/auprc/bacc/ece/nll/far...).
    """
    cfg = qcfg.base
    rng = np.random.default_rng(cfg.seed + hash(s_test) % 10_000)
    uniq = np.array(sorted(set(subjects)))
    train = np.array([u for u in uniq if u != s_test])
    # split train -> fit / calibration BY SUBJECT
    rng.shuffle(train)
    n_cal = max(1, int(round(cfg.cal_frac * len(train))))
    cal_subj = set(train[:n_cal]); fit_subj = [u for u in train if u not in cal_subj]

    # encoder fit on FIT windows only (baseline+stress features)
    fit_mask = np.isin(subjects, fit_subj)
    encoder = SuperNodeEncoder(feats, cfg).fit(X[fit_mask])
    gamma = cfg.gamma      # pinned; label-free gamma-selection can be added later
    Q_eff = len(encoder.branches)   # ACTUAL angle width (may be < cfg.n_qubits)

    rows = []
    for cond in conditions:
        noisy = (cond == "noisy")
        reservoir = QuantumReservoir(qcfg, noisy=noisy, n_qubits=Q_eff)

        # FIT: reservoir features on fit subjects, train ridge readout
        Xtr, ytr = [], []
        for s in fit_subj:
            m = (subjects == s)
            ys = y_state[m]
            keep = np.isin(ys, ["BASE", "STRESS"])
            if keep.sum() < 6:
                continue
            th = encoder.transform_angles(X[m][keep], gamma=gamma)
            order = np.argsort(~(ys[keep] == "BASE"))          # baseline first
            th = th[order]
            yb = (ys[keep][order] == "STRESS").astype(int)
            Xr = reservoir.run_sequence(th)[qcfg.washout:]
            yb = yb[qcfg.washout:]
            if len(Xr):
                Xtr.append(Xr); ytr.append(yb)
        if not Xtr:
            continue
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        readout = RidgeReadout(qcfg.ridge_lambda).fit(Xtr, ytr)

        # CAL: baseline scores on calibration subjects -> conformal
        a_cal = []
        for s in cal_subj:
            m = (subjects == s)
            ys = y_state[m]
            keep = (ys == "BASE")
            if keep.sum() < 3:
                continue
            th = encoder.transform_angles(X[m][keep], gamma=gamma)
            Xr = reservoir.run_sequence(th)[qcfg.washout:]
            if len(Xr):
                a_cal.append(readout.score(Xr))
        if not a_cal:
            continue
        a_cal = np.concatenate(a_cal)

        # TEST: the held-out subject's rest->stress monitoring stream
        m = (subjects == s_test)
        ys = y_state[m]
        keep = np.isin(ys, ["BASE", "STRESS"])
        if keep.sum() < 6 or (ys[keep] == "STRESS").sum() < 2:
            continue
        th_all = encoder.transform_angles(X[m][keep], gamma=gamma)
        order = np.argsort(~(ys[keep] == "BASE"))
        th_all = th_all[order]
        is_stress = (ys[keep][order] == "STRESS").astype(int)
        Xr = reservoir.run_sequence(th_all)
        Xr, is_stress = Xr[qcfg.washout:], is_stress[qcfg.washout:]
        if len(np.unique(is_stress)) < 2:
            continue
        a_test = readout.score(Xr)

        # scores -> conformal p-values -> calibrated anomaly prob pi
        p, pi = conformal_pvalues(a_cal, a_test)
        rank = ranking_metrics(pi, is_stress)
        ece, nll, brier = calibration_metrics(pi, is_stress)
        latency, far = alarm_metrics(p, is_stress.astype(bool), cfg.alpha, cfg.debounce_m)

        arm = f"QRC:{cond}"
        rows.append(dict(arm=arm, subject=s_test, cond=cond,
                         auroc=rank["auroc"], auprc=rank["auprc"], bacc=rank["bacc"],
                         prevalence=rank["prevalence"], auprc_norm=rank["auprc_norm"],
                         ece=ece, nll=nll, brier=brier,
                         far=far, latency=(np.nan if latency is None else latency),
                         n_test=len(is_stress), n_features=reservoir.n_features))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QRC core self-test")
    ap.add_argument("--smoke", action="store_true", help="synthetic smoke test, no real data")
    ap.add_argument("--modality", default="hrv")
    a = ap.parse_args()
    if a.smoke:
        cfg = Config(modality=a.modality, n_qubits=5, synthetic=True, quick=True,
                     seed=42).finalize()
        X, feats, y_state, subjects = load_dataset(cfg)
        qcfg = QRCConfig(base=cfg)
        subj = sorted(set(subjects))
        t0 = time.time()
        rows = run_fold_qrc(subj[-1], X, feats, y_state, subjects, qcfg,
                            conditions=("clean", "noisy"))
        print(f"\nsmoke fold on {subj[-1]}: {len(rows)} arms in {time.time()-t0:.1f}s")
        for r in rows:
            print(f"  {r['arm']:12s} AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f} "
                  f"bACC={r['bacc']:.3f} ECE={r['ece']:.3f} FAR={r['far']:.3f} "
                  f"nfeat={r['n_features']}")
        print("\nOK -- reservoir drives, readout trains, conformal+alarm+calibration reused.")