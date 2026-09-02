"""
noise_validation.py — validation of the model-adequacy diagnostic under noise.

Why this is needed
------------------
Section 3.11 reports that diastolic decay accelerates within the beat in a
median of 62% (arterial) and 75% (plethysmographic) of beats per case, and
uses this to conclude that the mono-exponential model does not hold. The
diagnostic operates on a numerical derivative, and differentiation amplifies
high-frequency content. If noise alone can produce apparent acceleration, part
of that figure is artefact rather than physiology.

The diagnostic has so far been validated only on synthetic Windkessel
waveforms at 45 dB SNR, where it correctly reports zero accelerating beats.
That establishes it does not fire under favourable conditions. It does not
establish its behaviour at the noise levels of clinical recordings.

What this script does
---------------------
A. NOISE SWEEP. Synthetic WK2 waveforms, where the mono-exponential model
   holds by construction, are corrupted across a range of SNR. Any beat
   flagged as accelerating is by definition a FALSE POSITIVE. Reports
   per-beat and per-case false-positive rates against SNR.

B. FILTER SENSITIVITY. Savitzky-Golay window length and polynomial order are
   swept at a fixed realistic SNR, to establish whether the reported rate
   depends on smoothing choices.

C. CLINICAL SNR ESTIMATE. Estimates the SNR of the real recordings, so the
   sweep in A can be read at the operating point that actually applies.
   Uses the residual of the fitted decay as a noise proxy.

D. CORRECTED FIGURE. Given the per-beat false-positive rate at clinical SNR,
   reports what fraction of the observed acceleration is attributable to
   noise, and gives a corrected estimate with its uncertainty.

Run:
    !python noise_validation.py --csv /content/drive/MyDrive/ppg_ident/results_c4.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from wk_sim import synth_case
from vitaldb_pipeline import detect_feet, load_waveforms, segment_quality, FS
from tau_fix import diastole_start
from tau_multibeat import _collect_tails

SPLIT_TOL = 1.30


# --------------------------------------------------------------------------
def accel_fraction(tails, fs=FS, win_s=0.04, poly=3, min_pts=60, trim=0.05):
    """Fraction of beats whose log-derivative slope is positive in either half.

    A positive slope means the decay rate increases within diastole, which no
    exponential can produce. On data where the model holds, any such beat is a
    false positive.
    """
    n_acc = n_tot = 0
    for tl in tails:
        if tl.size < min_pts:
            continue
        w = min(int(win_s * fs) | 1, (tl.size // 2) * 2 - 1)
        if w < poly + 2:
            continue
        try:
            sm = savgol_filter(np.asarray(tl, float), w, poly)
        except Exception:
            continue
        d = np.gradient(sm, 1.0 / fs)
        t = np.arange(tl.size) / fs
        m = d < 0
        if m.sum() < 30:
            continue
        tt, ln = t[m], np.log(-d[m])
        k = max(1, int(trim * tt.size))
        tt, ln = tt[k:-k], ln[k:-k]
        if tt.size < 24:
            continue
        h = tt.size // 2
        n_tot += 1
        for a, b in ((tt[:h], ln[:h]), (tt[h:], ln[h:])):
            A = np.vstack([a, np.ones_like(a)]).T
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            if c[0] >= 0:
                n_acc += 1
                break
    return (n_acc / n_tot if n_tot else np.nan), n_tot


def synth_tails(tau, snr_db, hr=72, rng=None, n_beats=8):
    d = synth_case(1.1, tau / 1.1, level="L0", gain=1.0, offset=0.0, hr=hr,
                   rng=rng, snr_db=snr_db, fs=FS, n_beats=20, keep_last=14)
    return _collect_tails(d["P"], detect_feet(d["P"], FS), diastole_start,
                          FS, n_beats=n_beats)


# ==========================================================================
def part_a(n_cases=40, seed=0):
    print("=" * 78)
    print("A. FALSE-POSITIVE RATE vs SNR  (model holds by construction)")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    snrs = [50, 45, 40, 35, 30, 25, 20, 15, 10]
    rows = []
    print(f"{'SNR (dB)':>9} {'beats':>7} {'per-beat FP':>13} "
          f"{'per-case FP (any beat)':>24}")
    for snr in snrs:
        fracs, tots = [], 0
        for _ in range(n_cases):
            tau = float(rng.uniform(1.0, 2.2))
            hr = float(rng.uniform(60, 85))
            tails = synth_tails(tau, snr, hr, rng)
            if len(tails) < 4:
                continue
            f, n = accel_fraction(tails)
            if np.isfinite(f):
                fracs.append(f); tots += n
        if not fracs:
            continue
        fracs = np.array(fracs)
        per_beat = float(np.mean(fracs))
        per_case = float(np.mean(fracs > 0))
        rows.append(dict(snr=snr, per_beat=per_beat, per_case=per_case,
                         beats=tots))
        print(f"{snr:9d} {tots:7d} {100*per_beat:12.1f}% "
              f"{100*per_case:23.1f}%")
    print("\n   Any flagged beat here is a false positive: the simulated")
    print("   waveform is mono-exponential by construction.")
    return pd.DataFrame(rows)


def part_b(snr_db=30, n_cases=30, seed=1):
    print("\n" + "=" * 78)
    print(f"B. SENSITIVITY TO SMOOTHING  (SNR = {snr_db} dB)")
    print("=" * 78)
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_cases):
        tau = float(rng.uniform(1.0, 2.2))
        tails = synth_tails(tau, snr_db, float(rng.uniform(60, 85)), rng)
        if len(tails) >= 4:
            cases.append(tails)
    print(f"{'window (ms)':>12} {'poly':>5} {'per-beat FP':>13}")
    out = []
    for win_s in (0.02, 0.03, 0.04, 0.06, 0.08):
        for poly in (2, 3, 4):
            fr = [accel_fraction(t, win_s=win_s, poly=poly)[0] for t in cases]
            fr = [x for x in fr if np.isfinite(x)]
            if not fr:
                continue
            v = float(np.mean(fr))
            out.append((win_s, poly, v))
            print(f"{1000*win_s:12.0f} {poly:5d} {100*v:12.1f}%")
    if out:
        vals = [v for _, _, v in out]
        print(f"\n   spread across settings: {100*min(vals):.1f}% to "
              f"{100*max(vals):.1f}%")
        print("   (the default used throughout is 40 ms, order 3)")
    return out


def part_c(n_cases=60, seed=2):
    """Estimate clinical SNR from the residual of the fitted decay."""
    print("\n" + "=" * 78)
    print("C. SIGNAL-TO-NOISE RATIO OF THE CLINICAL RECORDINGS")
    print("=" * 78)
    import pandas as pd
    trks = pd.read_csv("https://api.vitaldb.net/trks")
    ids = sorted(set(trks[trks.tname == "SNUADC/ART"].caseid) &
                 set(trks[trks.tname == "SNUADC/PLETH"].caseid))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), min(n_cases * 3, len(ids)), replace=False)

    res = {"abp": [], "ppg": []}
    got = 0
    for j in pick:
        if got >= n_cases:
            break
        cid = ids[j]
        try:
            v = load_waveforms(cid, fs=FS)
        except Exception:
            continue
        if v is None or v.shape[0] < FS * 720:
            continue
        mid = v.shape[0] // 2; half = int(FS * 60)
        abp, ppg = v[mid - half:mid + half, 0], v[mid - half:mid + half, 1]
        ok, _ = segment_quality(abp, ppg, FS)
        if not ok:
            continue
        abp = np.nan_to_num(abp, nan=float(np.nanmedian(abp)))
        ppg = np.nan_to_num(ppg, nan=float(np.nanmedian(ppg)))
        for sig, key in ((abp, "abp"), (ppg, "ppg")):
            tails = _collect_tails(sig, detect_feet(sig, FS), diastole_start,
                                   FS, n_beats=8)
            for tl in tails:
                if tl.size < 60:
                    continue
                w = min(int(0.04 * FS) | 1, (tl.size // 2) * 2 - 1)
                if w < 5:
                    continue
                sm = savgol_filter(np.asarray(tl, float), w, 3)
                noise = np.std(tl - sm)
                amp = np.ptp(sm)
                if noise > 0 and amp > 0:
                    res[key].append(20 * np.log10(amp / noise))
        got += 1

    for key, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        v = np.array(res[key])
        if v.size:
            print(f"   {lab:16s} SNR {np.median(v):5.1f} dB "
                  f"[{np.percentile(v,25):.1f}, {np.percentile(v,75):.1f}]  "
                  f"n_beats = {v.size}")
    print("\n   Read the table in part A at these values.")
    return {k: (float(np.median(v)) if len(v) else np.nan)
            for k, v in res.items()}


def part_d(fp_table, snr_clin, csv):
    print("\n" + "=" * 78)
    print("D. CORRECTED ESTIMATE")
    print("=" * 78)
    try:
        df = pd.read_csv(csv)
        ok = df[df.loaded == True]
        obs = {"abp": float(pd.to_numeric(ok.abp_accel, errors="coerce").median()),
               "ppg": float(pd.to_numeric(ok.ppg_accel, errors="coerce").median())}
    except Exception as e:
        print("   could not read observed rates:", e)
        return
    for key, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        s = snr_clin.get(key, np.nan)
        if not np.isfinite(s):
            continue
        fp = float(np.interp(s, fp_table.snr[::-1], fp_table.per_beat[::-1]))
        o = obs[key]
        corrected = max(0.0, (o - fp) / max(1e-9, 1 - fp))
        print(f"   {lab:16s} observed {100*o:5.1f}%   "
              f"false-positive at {s:.0f} dB {100*fp:5.1f}%   "
              f"corrected {100*corrected:5.1f}%")
    print("\n   Correction assumes false positives are independent of true")
    print("   acceleration, which is conservative: real curvature makes a")
    print("   beat more likely to be flagged, not less.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="results_c4.csv, for the observed acceleration rates")
    ap.add_argument("--skip-clinical", action="store_true")
    a = ap.parse_args()

    tbl = part_a()
    part_b()
    if not a.skip_clinical:
        snr = part_c()
        if a.csv:
            part_d(tbl, snr, a.csv)
