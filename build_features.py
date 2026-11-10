"""
================================================================================
 build_features.py -- WESAD (chest) + AffectiveROAD (wrist IoT) -> feature CSVs

 One extractor, two datasets (SWELL removed). Produces per-window feature tables
 consumed by the QRC pipeline / forecasting engine:

     WESAD  (chest RespiBAN, study-protocol labels):
         Dataset/wesad_hrv.csv   (ECG -> HRV)
         Dataset/wesad_eda.csv   (EDA)
         Dataset/wesad_resp.csv  (respiration waveform -> RESP)

     AffectiveROAD (E4 wrist, continuous 0..1 subjective stress target):
         Dataset/road_hrv.csv    (BVP/PPG -> HRV, pulse-rate variability)
         Dataset/road_eda.csv    (EDA)
         (no road_resp.csv: the distributed Bioharness has only 1 Hz breathing
          RATE, no respiration waveform -> a RESP feature cell is NOT derivable.)

 SHARED, per design: identical 60 s/5 s windowing and identical hrv_features()/
 eda_features(), so WESAD and ROAD cells are directly comparable. NO scaling /
 NO imputation here (fit per-fold on train subjects only); non-finite -> NaN.

 HONEST SIGNAL-QUALITY NOTE (state in the paper): ROAD HRV is wrist PPG under
 driving motion, so short-term variability (RMSSD/SDRR) is inflated by motion
 artifact relative to WESAD chest ECG. Use --motion-drop and the per-window
 `motion`/`n_beats` columns to gate. ROAD HRV is pulse-rate variability, not
 ECG HRV -- report it as such.

 LEAKAGE: ROAD LOSO must fold on driver_id (Records.txt: some drivers drove
 several drives), never on a single drive. WESAD folds on subject.

 USAGE:
   python build_features.py --wesad-root "WESAD/WESAD" --out Dataset
   python build_features.py --road-root  ".../AffectiveROAD_Data/.../Database" \
                            --wrist Left --motion-drop 3.0 --out Dataset
   python build_features.py --selftest
================================================================================
"""

from __future__ import annotations
import argparse, glob, os, pickle, re, struct, sys, warnings
import numpy as np
import pandas as pd
from scipy import signal, interpolate
from scipy.stats import kurtosis, skew
import zipfile as _zip
import csv as _csv

warnings.filterwarnings("ignore")

try:
    import neurokit2 as nk
    _HAS_NK = True
except Exception:
    _HAS_NK = False

# cfg
WIN_SEC   = 60.0          # window length
HOP_SEC   = 5.0           # window hop  -> 12 windows / minute
MIN_BEATS = 20            # min R-peaks in a window to compute HRV freq features
RR_MIN_MS, RR_MAX_MS = 300.0, 2000.0     # physiological RR gate (30-200 bpm)
EDA_FS    = 32.0          # EDA processing rate (slow signal; downsample to this)
RESP_FS   = 10.0          # respiration processing rate (very slow signal)

# WESAD label codes (from wesad_readme.pdf):
WESAD_LABELS = {1: "baseline", 2: "stress", 3: "amusement"}

# Signal Utilities
def _butter(x, fs, lo=None, hi=None, order=4):
    nyq = 0.5 * fs
    if lo and hi:
        b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="band")
    elif hi:
        b, a = signal.butter(order, hi / nyq, btype="low")
    elif lo:
        b, a = signal.butter(order, lo / nyq, btype="high")
    else:
        return x
    return signal.filtfilt(b, a, x, method="gust")

def _resample_to(x, fs_in, fs_out):
    if abs(fs_in - fs_out) < 1e-6:
        return np.asarray(x, float)
    n_out = int(round(len(x) * fs_out / fs_in))
    return signal.resample(np.asarray(x, float), max(n_out, 1))

# R-peak finding
def _pan_tompkins(ecg, fs):
    """Lightweight Pan-Tompkins-style detector (fallback when neurokit2 absent)."""
    x = _butter(ecg, fs, lo=5, hi=15, order=2)
    d = np.ediff1d(x, to_begin=0)
    sq = d * d
    win = max(1, int(0.150 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    thr = np.mean(integ) + 0.5 * np.std(integ)
    refractory = int(0.25 * fs)                       # 200-250 ms blanking
    peaks, last = [], -refractory
    cand = np.where(integ > thr)[0]
    for i in cand:
        if i - last < refractory:
            continue
        a, b = max(0, i - win), min(len(ecg), i + win)
        r = a + int(np.argmax(ecg[a:b]))              # snap to local ECG max
        if not peaks or r - peaks[-1] >= refractory:
            peaks.append(r); last = r
    return np.asarray(peaks, int)

def detect_rpeaks(ecg, fs):
    ecg = np.asarray(ecg, float).ravel()
    if _HAS_NK:
        try:
            ecg_c = nk.ecg_clean(ecg, sampling_rate=fs)
            _, info = nk.ecg_peaks(ecg_c, sampling_rate=fs)
            return np.asarray(info["ECG_R_Peaks"], int)
        except Exception:
            pass
    return _pan_tompkins(ecg, fs)

def rr_from_ecg(ecg, fs):
    """ECG -> cleaned RR-interval series in milliseconds."""
    pk = detect_rpeaks(ecg, fs)
    if len(pk) < 3:
        return np.array([])
    rr = np.diff(pk) / fs * 1000.0
    rr = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]             # physiological gate
    if len(rr) < 3:
        return rr
    med = np.median(rr)                                       # ectopic/jump removal
    rr = rr[np.abs(rr - med) <= 0.33 * med + 1e-9]
    return rr

# HRV Features
def _sampen(x, m=2, r=0.2):
    x = np.asarray(x, float); N = len(x)
    if N < m + 2:
        return np.nan
    r *= np.std(x)
    if r <= 0:
        return np.nan
    def _phi(mm):
        tmpl = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        C = 0
        for i in range(len(tmpl)):
            d = np.max(np.abs(tmpl - tmpl[i]), axis=1)
            C += np.sum(d <= r) - 1
        return C
    A, B = _phi(m + 1), _phi(m)
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan

def _higuchi(x, kmax=10):
    x = np.asarray(x, float); N = len(x)
    if N < 2 * kmax:
        return np.nan
    Lk = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lm.append(np.sum(np.abs(np.diff(x[idx]))) * (N - 1) / (len(idx) - 1) / k)
        if Lm:
            Lk.append(np.mean(Lm))
    if len(Lk) < 2:
        return np.nan
    lnk = np.log(1.0 / np.arange(1, len(Lk) + 1))
    return float(np.polyfit(lnk, np.log(Lk), 1)[0])

def _psd_bands(rr_ms):
    """Welch PSD on a 4 Hz cubic-interpolated RR tachogram -> band powers."""
    t = np.cumsum(rr_ms) / 1000.0
    if len(t) < 4 or t[-1] - t[0] < 20:
        return {k: np.nan for k in ("VLF", "LF", "HF", "TP")}
    fs_i = 4.0
    tt = np.arange(t[0], t[-1], 1 / fs_i)
    f = interpolate.interp1d(t, rr_ms, kind="cubic", fill_value="extrapolate")(tt)
    f = f - np.mean(f)
    fr, ps = signal.welch(f, fs=fs_i, nperseg=min(256, len(f)))
    band = lambda lo, hi: float(np.trapezoid(ps[(fr >= lo) & (fr < hi)],
                                             fr[(fr >= lo) & (fr < hi)]))
    vlf, lf, hf = band(0.0033, 0.04), band(0.04, 0.15), band(0.15, 0.40)
    return {"VLF": vlf, "LF": lf, "HF": hf, "TP": vlf + lf + hf}

def hrv_features(rr_ms):
    rr = np.asarray(rr_ms, float)
    out = {}
    if len(rr) < 3:
        return out
    d = np.diff(rr)
    out["MEAN_RR"]   = float(np.mean(rr))
    out["MEDIAN_RR"] = float(np.median(rr))
    out["SDRR"]      = float(np.std(rr, ddof=1))
    out["RMSSD"]     = float(np.sqrt(np.mean(d ** 2)))
    out["SDSD"]      = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
    out["SDRR_RMSSD"] = out["SDRR"] / out["RMSSD"] if out["RMSSD"] else np.nan
    out["HR"]        = float(60000.0 / np.mean(rr))
    out["pNN25"]     = float(np.mean(np.abs(d) > 25) * 100)
    out["pNN50"]     = float(np.mean(np.abs(d) > 50) * 100)
    # Poincare
    out["SD1"] = float(np.sqrt(0.5) * np.std(d, ddof=1)) if len(d) > 1 else np.nan
    out["SD2"] = float(np.sqrt(max(2 * out["SDRR"] ** 2 - 0.5 * np.std(d, ddof=1) ** 2, 0))) \
        if len(d) > 1 else np.nan
    # distribution / nonlinear
    out["KURT"] = float(kurtosis(rr)); out["SKEW"] = float(skew(rr))
    out["sampen"] = _sampen(rr); out["higuci"] = _higuchi(rr)
    # relative-RR branch (relativeRR)
    rel = 2.0 * d / (rr[:-1] + rr[1:] + 1e-9)
    if len(rel) > 1:
        out["MEAN_REL_RR"]   = float(np.mean(rel))
        out["MEDIAN_REL_RR"] = float(np.median(rel))
        out["SDRR_REL_RR"]   = float(np.std(rel, ddof=1))
        out["RMSSD_REL_RR"]  = float(np.sqrt(np.mean(np.diff(rel) ** 2))) if len(rel) > 1 else np.nan
    # frequency domain (sympathovagal branch)
    if len(rr) >= MIN_BEATS:
        b = _psd_bands(rr)
        out.update(b)
        tp = b["TP"] or np.nan
        out["VLF_PCT"] = 100 * b["VLF"] / tp if tp else np.nan
        out["LF_PCT"]  = 100 * b["LF"] / tp if tp else np.nan
        out["HF_PCT"]  = 100 * b["HF"] / tp if tp else np.nan
        lfhf = b["LF"] + b["HF"]
        out["LF_NU"] = 100 * b["LF"] / lfhf if lfhf else np.nan
        out["HF_NU"] = 100 * b["HF"] / lfhf if lfhf else np.nan
        out["LF_HF"] = b["LF"] / b["HF"] if b["HF"] else np.nan
        out["HF_LF"] = b["HF"] / b["LF"] if b["LF"] else np.nan
    return out

# EDA Features
def _tonic_phasic(eda, fs):
    if _HAS_NK:
        try:
            d = nk.eda_phasic(nk.eda_clean(eda, sampling_rate=fs), sampling_rate=fs)
            return d["EDA_Tonic"].values, d["EDA_Phasic"].values
        except Exception:
            pass
    tonic = _butter(eda, fs, hi=0.05, order=2)        # slow drive
    phasic = eda - tonic
    return tonic, phasic

def eda_features(eda, fs):
    eda = np.asarray(eda, float).ravel()
    out = {}
    if len(eda) < int(2 * fs):
        return out
    tonic, phasic = _tonic_phasic(eda, fs)
    # tonic (SCL)
    out["SCL_MEAN"]   = float(np.mean(tonic))
    out["SCL_MEDIAN"] = float(np.median(tonic))
    out["TONIC_SLOPE"] = float(np.polyfit(np.arange(len(tonic)), tonic, 1)[0])
    # variability
    out["EDA_STD"]   = float(np.std(eda, ddof=1))
    out["EDA_RANGE"] = float(np.ptp(eda))
    out["EDA_MIN"]   = float(np.min(eda)); out["EDA_MAX"] = float(np.max(eda))
    out["EDA_IQR"]   = float(np.subtract(*np.percentile(eda, [75, 25])))
    out["EDA_SLOPE"] = float(np.polyfit(np.arange(len(eda)), eda, 1)[0])
    # phasic peaks -> responserate + shape + phasic amplitude
    ph = phasic - np.median(phasic)
    height = 0.01 + 0.5 * np.std(ph)                  # ~0.01 muS floor
    pk, props = signal.find_peaks(ph, height=height, distance=int(1.0 * fs))
    dur_min = len(eda) / fs / 60.0
    out["NSCR"]     = float(len(pk))
    out["SCR_RATE"] = float(len(pk) / dur_min) if dur_min > 0 else np.nan
    if len(pk):
        amps = props["peak_heights"]
        out["SCR_AMP_MEAN"] = float(np.mean(amps))
        out["SCR_AMP_MAX"]  = float(np.max(amps))
        out["PHASIC_AUC"]   = float(np.trapezoid(np.clip(ph, 0, None)) / fs)
        # rise time: samples from preceding trough to peak
        rises = []
        for p in pk:
            a = max(0, p - int(5 * fs))
            tr = a + int(np.argmin(ph[a:p + 1])) if p > a else a
            rises.append((p - tr) / fs)
        out["SCR_RISETIME_MEAN"] = float(np.mean(rises)) if rises else np.nan
        out["SCR_HALF_RECOVERY"] = float(np.median(np.diff(pk)) / fs) if len(pk) > 1 else np.nan
    else:
        for k in ("SCR_AMP_MEAN", "SCR_AMP_MAX", "PHASIC_AUC",
                  "SCR_RISETIME_MEAN", "SCR_HALF_RECOVERY"):
            out[k] = np.nan
    # nonlinear
    out["EDA_KURT"] = float(kurtosis(eda)); out["EDA_SKEW"] = float(skew(eda))
    out["EDA_SAMPEN"] = _sampen(_resample_to(eda, fs, 4.0))      # cheap on 4 Hz copy
    return out

# RESP Features  (WESAD-only modality)        
def _resp_branch(col):
    """
    Group a respiration feature into a super-node (one per qubit).
    Mirror of qrc_base._hrv_branch / _eda_branch; checks most-specific first.
    Branches: rate / interval / variability / amplitude / shape / nonlinear.
    """
    u = col.upper()
    if any(k in u for k in ["KURT", "SKEW", "SAMPEN", "ENTROP", "HIGUC"]):   return "nonlinear"
    if any(k in u for k in ["IE_RATIO", "INHALE", "EXHALE", "INSP", "EXP",
                            "RISE", "DUTY"]):                                return "shape"
    if any(k in u for k in ["AMP", "DEPTH", "EXCURSION"]):                  return "amplitude"
    if any(k in u for k in ["STD", "_SD", "SD_", "VAR", "RMSSD", "RANGE",
                            "IQR", "RRV", "_CV"]):                          return "variability"
    if any(k in u for k in ["BBI", "INTERVAL", "PERIOD", "_BB"]):           return "interval"
    if any(k in u for k in ["RATE", "FREQ", "BREATH", "DOM_"]):            return "rate"
    return "other"

def resp_features(resp, fs):
    """
    Respiration belt window -> breathing features.

    fs here is RESP_FS (window_signal resamples before calling). Extracts
    breathing rate, breath-to-breath interval + its variability (RRV, the
    breathing analogue of HRV), tidal amplitude, inhale/exhale timing (I:E),
    dominant respiratory frequency, and distribution/nonlinear shape terms.
    """
    resp = np.asarray(resp, float).ravel()
    out = {}
    if len(resp) < int(5 * fs):
        return out
    x = _butter(resp, fs, lo=0.1, hi=0.6, order=2)          # 6-36 breaths/min band
    x = x - np.mean(x)
    dist = int(1.2 * fs)                                     # >= ~1.2 s between breaths
    h = 0.3 * np.std(x)
    peaks, _ = signal.find_peaks(x, distance=dist, height=h)        # inhalation peaks
    troughs, _ = signal.find_peaks(-x, distance=dist, height=h)     # exhalation troughs
    dur_min = len(resp) / fs / 60.0
    out["RESP_RATE"] = float(len(peaks) / dur_min) if dur_min > 0 else np.nan

    if len(peaks) >= 3:                                      # interval + variability
        bbi = np.diff(peaks) / fs * 1000.0                  # breath-to-breath, ms
        out["BBI_MEAN"]   = float(np.mean(bbi))
        out["BBI_MEDIAN"] = float(np.median(bbi))
        out["BBI_STD"]    = float(np.std(bbi, ddof=1))
        out["BBI_RMSSD"]  = float(np.sqrt(np.mean(np.diff(bbi) ** 2))) if len(bbi) > 1 else np.nan
        out["RRV_CV"]     = float(np.std(bbi, ddof=1) / np.mean(bbi)) if np.mean(bbi) else np.nan
    else:
        for k in ("BBI_MEAN", "BBI_MEDIAN", "BBI_STD", "BBI_RMSSD", "RRV_CV"):
            out[k] = np.nan

    amps = [x[p] - x[troughs[troughs < p][-1]]              # tidal amplitude
            for p in peaks if (troughs < p).any()]
    out["AMP_MEAN"] = float(np.mean(amps)) if amps else np.nan
    out["AMP_MAX"]  = float(np.max(amps)) if amps else np.nan

    inh = [(p - troughs[troughs < p][-1]) / fs for p in peaks if (troughs < p).any()]
    exh = [(troughs[troughs > p][0] - p) / fs for p in peaks if (troughs > p).any()]
    out["INHALE_DUR"] = float(np.mean(inh)) if inh else np.nan
    out["EXHALE_DUR"] = float(np.mean(exh)) if exh else np.nan
    out["IE_RATIO"]   = (float(np.mean(inh) / np.mean(exh))
                         if inh and exh and np.mean(exh) else np.nan)

    fr, ps = signal.welch(x, fs=fs, nperseg=min(len(x), int(50 * fs)))
    band = (fr >= 0.1) & (fr <= 0.6)
    out["DOM_FREQ"] = float(fr[band][np.argmax(ps[band])]) if band.any() and ps[band].sum() else np.nan

    out["RESP_KURT"]   = float(kurtosis(resp))
    out["RESP_SKEW"]   = float(skew(resp))
    out["RESP_SAMPEN"] = _sampen(_resample_to(resp, fs, 4.0))
    return out

# Windowing
def _contiguous_runs(labels):
    """Yield (start, end, label) for maximal runs of a constant label."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return
    change = np.where(np.diff(labels) != 0)[0] + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield int(a), int(b), labels[a]

def window_signal(sig, fs, label_vec, keep_labels, feat_fn, fs_target=None,
                  subject=None, label_names=None):
    """Slide WIN_SEC/HOP_SEC windows within each constant-label run; one row each."""
    rows = []
    win = int(WIN_SEC * fs); hop = int(HOP_SEC * fs)
    for a, b, lab in _contiguous_runs(label_vec):
        if lab not in keep_labels or (b - a) < win:
            continue
        cond = label_names[lab] if label_names else str(lab)
        for s in range(a, b - win + 1, hop):
            seg = sig[s:s + win]
            x = _resample_to(seg, fs, fs_target) if fs_target else seg
            feat = feat_fn(x, fs_target or fs)
            if feat:
                rows.append({"subject": subject, "condition": cond, **feat})
    return rows

# WESAD
def load_wesad_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    sid = int(re.sub(r"\D", "", str(d["subject"])))
    lab = np.asarray(d["label"]).ravel().astype(int)
    ecg = np.asarray(d["signal"]["chest"]["ECG"]).ravel()       # 700 Hz
    eda = np.asarray(d["signal"]["chest"]["EDA"]).ravel()       # 700 Hz
    resp = np.asarray(d["signal"]["chest"]["Resp"]).ravel()     # 700 Hz
    return sid, lab, ecg, eda, resp

def build_wesad(root, out_dir):
    pkls = sorted(glob.glob(os.path.join(root, "S*", "S*.pkl")))
    if not pkls:
        print(f"[wesad] no pkl under {root}; skipping."); return
    fs = 700.0
    hrv_rows, eda_rows, resp_rows = [], [], []
    for p in pkls:
        sid, lab, ecg, eda, resp = load_wesad_pkl(p)
        def hrv_fn(seg, _fs):  # ECG window -> RR -> HRV
            return hrv_features(rr_from_ecg(seg, fs))
        hrv_rows += window_signal(ecg, fs, lab, set(WESAD_LABELS), hrv_fn,
                                  subject=sid, label_names=WESAD_LABELS)
        eda_rows += window_signal(eda, fs, lab, set(WESAD_LABELS), eda_features,
                                  fs_target=EDA_FS, subject=sid, label_names=WESAD_LABELS)
        resp_rows += window_signal(resp, fs, lab, set(WESAD_LABELS), resp_features,
                                   fs_target=RESP_FS, subject=sid, label_names=WESAD_LABELS)
        print(f"[wesad] S{sid}: {len(hrv_rows)} hrv / {len(eda_rows)} eda / "
              f"{len(resp_rows)} resp rows (cum.)")
    _write(hrv_rows, out_dir, "wesad_hrv.csv")
    _write(eda_rows, out_dir, "wesad_eda.csv")
    _write(resp_rows, out_dir, "wesad_resp.csv")

# IO + QC
def _write(rows, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="all", subset=[c for c in df.columns
                                              if c not in ("subject", "condition")])
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    by = df.groupby("condition").size().to_dict() if "condition" in df else {}
    print(f"  -> {path}: {len(df)} windows, "
          f"{df.shape[1]-2} feats, {df['subject'].nunique()} subjects, states={by}")

# SELF-TEST (synthetic; validates the DSP core without any dataset)
def selftest():
    rng = np.random.default_rng(0); fs = 700.0; T = 180
    t = np.arange(0, T, 1 / fs)
    # synthetic ECG: QRS spikes at ~ a drifting heart rate + noise
    hr = 70 + 10 * np.sin(2 * np.pi * t / 120)
    phase = np.cumsum(hr / 60.0 / fs)
    ecg = signal.windows.gaussian(int(0.05 * fs), std=int(0.008 * fs))
    spikes = np.zeros_like(t)
    beat_idx = np.where(np.diff(np.floor(phase)) > 0)[0]
    for b in beat_idx:
        s = b - len(ecg) // 2
        if 0 <= s < len(spikes) - len(ecg):
            spikes[s:s + len(ecg)] += ecg
    ecg_sig = spikes + 0.02 * rng.standard_normal(len(t))
    # synthetic EDA: tonic drift + a few SCRs
    eda = 5 + 0.5 * np.sin(2 * np.pi * t / 300)
    for c in rng.integers(0, len(t), 8):
        k = np.exp(-np.arange(int(8 * fs)) / (2 * fs))
        if c + len(k) < len(eda):
            eda[c:c + len(k)] += rng.uniform(0.2, 1.0) * k

    # synthetic respiration: ~15 breaths/min sinusoid + harmonic + noise
    resp = (np.sin(2 * np.pi * 0.25 * t) + 0.3 * np.sin(2 * np.pi * 0.5 * t)
            + 0.05 * rng.standard_normal(len(t)))

    lab = np.zeros(len(t), int)
    lab[: len(t)//3] = 1; lab[len(t)//3: 2*len(t)//3] = 2; lab[2*len(t)//3:] = 3

    hrv_rows = window_signal(ecg_sig, fs, lab, set(WESAD_LABELS),
                             lambda s, _f: hrv_features(rr_from_ecg(s, fs)),
                             subject=99, label_names=WESAD_LABELS)
    eda_rows = window_signal(eda, fs, lab, set(WESAD_LABELS), eda_features,
                             fs_target=EDA_FS, subject=99, label_names=WESAD_LABELS)
    resp_rows = window_signal(resp, fs, lab, set(WESAD_LABELS), resp_features,
                              fs_target=RESP_FS, subject=99, label_names=WESAD_LABELS)
    h = pd.DataFrame(hrv_rows); e = pd.DataFrame(eda_rows); r = pd.DataFrame(resp_rows)
    print(f"[selftest] HRV windows={len(h)} cols={[c for c in h.columns][:8]}...")
    print(f"[selftest] sample HR={h['HR'].median():.1f} bpm  RMSSD={h['RMSSD'].median():.1f} ms "
          f"LF/HF={h['LF_HF'].median():.2f}")
    print(f"[selftest] EDA windows={len(e)} NSCR_med={e['NSCR'].median()} "
          f"SCL_MEAN_med={e['SCL_MEAN'].median():.2f}")
    print(f"[selftest] RESP windows={len(r)} rate_med={r['RESP_RATE'].median():.1f}/min "
          f"IE_RATIO_med={r['IE_RATIO'].median():.2f} DOM_FREQ_med={r['DOM_FREQ'].median():.3f}Hz")
    assert {"MEAN_RR","RMSSD","LF_HF","HR"} <= set(h.columns), "HRV schema broken"
    assert {"SCL_MEAN","NSCR","SCR_RATE","EDA_KURT"} <= set(e.columns), "EDA schema broken"
    assert {"RESP_RATE","BBI_MEAN","AMP_MEAN","IE_RATIO","RESP_KURT"} <= set(r.columns), "RESP schema broken"
    assert h["HR"].between(40, 120).mean() > 0.8, "HR detection off"
    assert r["RESP_RATE"].between(8, 25).mean() > 0.8, "RESP rate detection off"
    rb = sorted({_resp_branch(c) for c in r.columns if c not in ("subject","condition")})
    print(f"[selftest] RESP branches={rb}")
    assert len(rb) >= 6, "RESP branch coverage < 6"
    print("[selftest] OK -- schema + detection sane.  neurokit2:", _HAS_NK)


# ROAD
E4_ANNOT_RATE = 4.0                     # Annot_E4_*.csv indices are 4 Hz samples
E4_DRIVE_COLS = slice(2, 18)            # 16 driving landmarks (skip lead/trail Rest)
SM_DRIVE_COLS = slice(0, 16)            # same 16 driving landmarks in SM annotation
MIN_HRV_BEATS = 30                      # min PPG beats in a 60 s window to accept HRV

def detect_ppg_peaks(bvp, fs):
    """Systolic-peak detector on E4 BVP (bandpass pulse band + refractory)."""
    x = _butter(np.asarray(bvp, float).ravel(), fs, lo=0.5, hi=8.0, order=2)
    dist = int(0.33 * fs)                               # <= ~180 bpm
    pk, _ = signal.find_peaks(x, distance=dist, height=np.median(x))
    return pk

def ppi_from_bvp(bvp, fs):
    """E4 BVP -> cleaned peak-to-peak interval series (ms), analogue of rr_from_ecg."""
    pk = detect_ppg_peaks(bvp, fs)
    if len(pk) < 3:
        return np.array([])
    ppi = np.diff(pk) / fs * 1000.0
    ppi = ppi[(ppi >= RR_MIN_MS) & (ppi <= RR_MAX_MS)]  # physiological gate
    if len(ppi) < 3:
        return ppi
    med = np.median(ppi)
    return ppi[np.abs(ppi - med) <= 0.33 * med + 1e-9]  # ectopic/jump removal

def _road_annot(path):
    out = {}
    for row in _csv.reader(open(path, newline="")):
        if row and re.match(r"(?i)^drv\d+$", row[0].strip()):
            out[row[0].strip()] = [int(float(x)) for x in row[1:] if x.strip() != ""]
    return out

def _e4_from_zip(zpath, name):
    """Empatica file: row1=unix start, row2=fs, then samples. ACC -> 3 cols."""
    with _zip.ZipFile(zpath) as z:
        raw = z.read(name).decode("utf-8", "ignore")
    rows = [r for r in raw.splitlines() if r.strip() != ""]
    if name == "ACC.csv":
        fs = float(rows[1].split(",")[0])
        data = np.array([[float(v) for v in r.split(",")] for r in rows[2:]], float)
        return data, fs
    fs = float(rows[1])
    return np.array([float(r) for r in rows[2:]], float), fs

def _road_driver_of(records_path):
    link = {}
    for line in open(records_path, encoding="latin1"):
        m = re.match(r"\s*Drv(\d+)\s*:", line)
        if m:
            dm = re.search(r"driver\s*(D\d+)", line)
            if dm:
                link[int(m.group(1))] = dm.group(1)
    return lambda d: link.get(d, f"solo{d}")

def _e4_zip_paths(e4dir, wrist):
    exact = os.path.join(e4dir, f"{wrist}.zip")
    if os.path.exists(exact):
        return [exact]
    return sorted(os.path.join(e4dir, z) for z in os.listdir(e4dir)
                  if re.match(rf"{wrist}\d+\.zip$", z))          # split recordings

def _window_motion(acc, fs_acc, w0_s, w1_s):
    if acc is None:
        return np.nan
    i0, i1 = int(w0_s * fs_acc), int(w1_s * fs_acc)
    seg = acc[max(0, i0):min(len(acc), i1)]
    return float(np.mean(np.linalg.norm(np.diff(seg, axis=0), axis=1))) if len(seg) > 1 else np.nan

def _find_road_db(root):
    """
    Locate the AffectiveROAD 'Database' dir (the one holding Records.txt +
    E4/ + Subj_metric/ + Bioharness/), even when --road-root points a level or
    two above it -- the zip nests AffectiveROAD_Data/AffectiveROAD_Data/Database.
    """
    def is_db(d):
        return (os.path.isfile(os.path.join(d, "Records.txt")) and
                os.path.isdir(os.path.join(d, "E4")) and
                os.path.isdir(os.path.join(d, "Subj_metric")))
    root = os.path.abspath(root)
    if not os.path.exists(root):
        raise FileNotFoundError(
            f"--road-root path does not exist: {root}\n"
            f"  (On Windows a LEADING '/' means the drive root, e.g. C:\\ -- drop it,\n"
            f"   or pass the full path like C:\\Users\\you\\Desktop\\...\\AffectiveROAD_Data)")
    if is_db(root):
        return root
    for cur, _dirs, _files in os.walk(root):          # search downward for Database
        if is_db(cur):
            print(f"  [road] found Database at: {cur}")
            return cur
    raise FileNotFoundError(
        f"Could not find the AffectiveROAD 'Database' folder under:\n    {root}\n"
        f"  It must contain: Records.txt, E4/, Subj_metric/, Bioharness/.\n"
        f"  Point --road-root at the folder that CONTAINS 'Database' (or at 'Database' itself).")

def build_road(root, out_dir, wrist="Left", motion_drop=None):
    """Build road_hrv.csv (BVP->HRV) and road_eda.csv (EDA) with continuous target."""
    root = _find_road_db(root)                        # robust path resolution
    driver_of = _road_driver_of(os.path.join(root, "Records.txt"))
    e4_annot  = _road_annot(os.path.join(root, "E4", f"Annot_E4_{wrist}.csv"))
    sm_annot  = _road_annot(os.path.join(root, "Subj_metric", "Annot_Subjective_metric.csv"))
    drives = sorted(e4_annot, key=lambda d: int(re.sub(r"\D", "", d)))
    hrv_rows, eda_rows = [], []
    for drv in drives:
        dnum = int(re.sub(r"\D", "", drv))
        e4dir = next((os.path.join(root, "E4", d) for d in os.listdir(os.path.join(root, "E4"))
                      if d.endswith(f"E4-{drv}")), None)
        if e4dir is None or drv not in e4_annot or drv not in sm_annot:
            print(f"  [road skip] {drv}: missing folder/annotation"); continue
        zpaths = _e4_zip_paths(e4dir, wrist)
        if not zpaths:
            print(f"  [road skip] {drv}: no {wrist}*.zip"); continue
        eda_p, bvp_p, acc_p, fs_eda, fs_bvp, fs_acc = [], [], [], None, None, None
        for zp in zpaths:
            e, fs_eda = _e4_from_zip(zp, "EDA.csv"); eda_p.append(e)
            b, fs_bvp = _e4_from_zip(zp, "BVP.csv"); bvp_p.append(b)
            try:
                a, fs_acc = _e4_from_zip(zp, "ACC.csv"); acc_p.append(a)
            except Exception:
                pass
        eda = np.concatenate(eda_p); bvp = np.concatenate(bvp_p)
        acc = np.concatenate(acc_p) if acc_p else None
        e4_lm = np.array(e4_annot[drv][E4_DRIVE_COLS], float)     # 4 Hz sample anchors
        sm_lm = np.array(sm_annot[drv][SM_DRIVE_COLS], float)     # SM sample anchors
        sm = np.loadtxt(os.path.join(root, "Subj_metric", f"SM_{drv}.csv"), skiprows=1)
        if len(e4_lm) != len(sm_lm) or len(e4_lm) < 2:
            print(f"  [road skip] {drv}: landmark mismatch"); continue
        driver = driver_of(dnum)

        def sm_at(e4_sample_4hz):                                 # landmark-anchored map
            j = int(round(np.interp(e4_sample_4hz, e4_lm, sm_lm)))
            return float(sm[min(max(j, 0), len(sm) - 1)])

        # EDA cell (4 Hz)
        win, hop = int(WIN_SEC * fs_eda), int(HOP_SEC * fs_eda)
        for s in range(int(e4_lm[0]), int(e4_lm[-1]) - win + 1, hop):
            feat = eda_features(eda[s:s + win], fs_eda)
            if not feat:
                continue
            end = s + win
            eda_rows.append({"driver_id": driver, "drive_id": drv,
                             "t_sec": round(end / fs_eda, 3),
                             "motion": _window_motion(acc, fs_acc, s / fs_eda, end / fs_eda),
                             "stress_target": sm_at(end), **feat})
        # HRV cell (BVP 64 Hz; annotation is 4 Hz so scale by fs_bvp/4)
        r = fs_bvp / E4_ANNOT_RATE                                # 64/4 = 16
        winb, hopb = int(WIN_SEC * fs_bvp), int(HOP_SEC * fs_bvp)
        for s in range(int(e4_lm[0] * r), int(e4_lm[-1] * r) - winb + 1, hopb):
            ppi = ppi_from_bvp(bvp[s:s + winb], fs_bvp)
            if len(ppi) < MIN_HRV_BEATS:                          # motion-degraded window
                continue
            feat = hrv_features(ppi)
            if not feat:
                continue
            end = s + winb
            hrv_rows.append({"driver_id": driver, "drive_id": drv,
                             "t_sec": round(end / fs_bvp, 3),
                             "motion": _window_motion(acc, fs_acc, s / fs_bvp, end / fs_bvp),
                             "n_beats": len(ppi),
                             "stress_target": sm_at(end / r), **feat})
        print(f"  [road {drv}] driver={driver}  EDA={sum(1 for x in eda_rows if x['drive_id']==drv)}"
              f"  HRV={sum(1 for x in hrv_rows if x['drive_id']==drv)} windows")

    for rows, name in [(hrv_rows, "road_hrv.csv"), (eda_rows, "road_eda.csv")]:
        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        if motion_drop is not None and "motion" in df:
            n0 = len(df); df = df[~(df["motion"] > motion_drop)]
            print(f"  [motion] {name}: dropped {n0 - len(df)}/{n0} windows > {motion_drop}")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, name), index=False)
        print(f"  -> {name}: {len(df)} windows, {df['driver_id'].nunique()} drivers, "
              f"{df['drive_id'].nunique()} drives")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wesad-root", default=None, help='WESAD S*/S*.pkl root')
    ap.add_argument("--road-root", default=None,
                    help="AffectiveROAD Database/ (Bioharness, E4, Subj_metric, Records.txt)")
    ap.add_argument("--wrist", default="Left", choices=["Left", "Right"],
                    help="E4 wrist to use for ROAD")
    ap.add_argument("--motion-drop", type=float, default=None,
                    help="drop windows whose ACC motion exceeds this (default: keep + flag)")
    ap.add_argument("--out", default="Dataset")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.wesad_root or a.road_root):
        ap.error("nothing to do: pass --wesad-root and/or --road-root (or --selftest)")
    if a.wesad_root:
        build_wesad(a.wesad_root, a.out)
    if a.road_root:
        build_road(a.road_root, a.out, wrist=a.wrist, motion_drop=a.motion_drop)

if __name__ == "__main__":
    main()