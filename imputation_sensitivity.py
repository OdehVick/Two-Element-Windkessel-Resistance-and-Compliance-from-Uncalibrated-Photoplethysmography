"""
imputation_sensitivity.py — item 2: does median imputation create false
acceleration?

The concern
-----------
The pipeline replaces missing samples with the channel median. That inserts
flat plateaus. The adequacy diagnostic then differentiates the signal, and at
the edge of a plateau the derivative changes abruptly; ln(-dP/dt) can turn
upward, which the diagnostic reads as accelerating decay. The quality gate
permits up to 2% missing samples, which over a 120 s window at 500 Hz is up to
1200 imputed points. The noise validation reported previously used synthetic
signals with no gaps and therefore could not detect this.

The test
--------
Each recording is analysed three ways, with the same diagnostic:

    baseline    missing samples replaced by the channel median (as published)
    excluded    any beat containing an imputed sample is discarded
    repaired    short gaps filled by shape-preserving interpolation, beats
                containing a long gap discarded

If the acceleration rate is essentially unchanged across the three, imputation
is not driving the result. If it falls sharply under 'excluded', the published
figure is contaminated and must be recomputed.

Run:
    !python imputation_sensitivity.py --n 400 \
        --out /content/drive/MyDrive/ppg_ident/results_imputation.csv
"""

import argparse, os, time
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from vitaldb_pipeline import load_waveforms, segment_quality, detect_feet, FS
from tau_fix import diastole_start
from tau_multibeat import _collect_tails

WIN_S, SKIP_START_S, N_BEATS = 120.0, 600.0, 8
MAX_GAP_S = 0.02          # 10 samples at 500 Hz; longer gaps are not repaired


# --------------------------------------------------------------------------
def runs_of_true(mask):
    """(start, length) of each contiguous True run."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out


def repair(sig, mask, fs=FS, max_gap_s=MAX_GAP_S):
    """Shape-preserving interpolation of short gaps. None if a gap is long."""
    if not mask.any():
        return sig.copy()
    lim = int(max_gap_s * fs)
    if any(l > lim for _, l in runs_of_true(mask)):
        return None
    good = ~mask
    if good.sum() < 10:
        return None
    f = PchipInterpolator(np.flatnonzero(good), sig[good], extrapolate=True)
    out = sig.copy()
    out[mask] = f(np.flatnonzero(mask))
    return out


def accel_fraction(tails, fs=FS, win_s=0.04, poly=3, trim=0.05, min_pts=60):
    """Fraction of beats with a positive log-derivative slope in either half."""
    n_acc = n_tot = 0
    for tl in tails:
        tl = np.asarray(tl, float)
        if tl.size < min_pts:
            continue
        w = min(int(win_s * fs) | 1, (tl.size // 2) * 2 - 1)
        if w < poly + 2:
            continue
        try:
            sm = savgol_filter(tl, w, poly)
        except Exception:
            continue
        dd = np.gradient(sm, 1.0 / fs)
        t = np.arange(tl.size) / fs
        m = dd < 0
        if m.sum() < 30:
            continue
        tt, ln = t[m], np.log(-dd[m])
        k = max(1, int(trim * tt.size))
        tt, ln = tt[k:-k], ln[k:-k]
        if tt.size < 24:
            continue
        n_tot += 1
        h = tt.size // 2
        for a, b in ((tt[:h], ln[:h]), (tt[h:], ln[h:])):
            A = np.vstack([a, np.ones_like(a)]).T
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            if c[0] >= 0:
                n_acc += 1
                break
    return (n_acc / n_tot if n_tot else np.nan), n_tot


def tails_with_mask(sig, mask, fs=FS, n_beats=N_BEATS):
    """Diastolic tails plus, for each, whether it contains an imputed sample."""
    feet = detect_feet(sig, fs)
    tails, flags = [], []
    for a, b in zip(feet[:-1], feet[1:]):
        if len(tails) >= n_beats:
            break
        beat = sig[a:b]
        i0 = diastole_start(beat, fs)
        if i0 is None:
            continue
        tl = beat[i0:]
        if tl.size < 60:
            continue
        tails.append(tl)
        flags.append(bool(mask[a + i0:b].any()))
    return tails, np.array(flags, dtype=bool)


# --------------------------------------------------------------------------
def process_one(caseid):
    rec = dict(caseid=int(caseid), ok=False, reason="")
    try:
        v = load_waveforms(caseid, fs=FS)
    except Exception as e:
        rec["reason"] = f"load:{type(e).__name__}"
        return rec
    if v is None or v.shape[0] < FS * (SKIP_START_S + WIN_S):
        rec["reason"] = "too short"
        return rec

    mid = v.shape[0] // 2
    half = int(FS * WIN_S / 2)
    abp_raw = v[mid - half:mid + half, 0]
    ppg_raw = v[mid - half:mid + half, 1]
    ok, reason = segment_quality(abp_raw, ppg_raw, FS)
    if not ok:
        rec["reason"] = reason
        return rec

    rec.update(ok=True, reason="ok")
    for raw, pre in ((abp_raw, "abp"), (ppg_raw, "ppg")):
        mask = ~np.isfinite(raw)
        rec[f"{pre}_miss_frac"] = float(mask.mean())

        # ---- baseline: median substitution, as published
        base = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
        tails, flags = tails_with_mask(base, mask)
        f_base, n_base = accel_fraction(tails)
        rec[f"{pre}_accel_base"] = f_base
        rec[f"{pre}_nbeats_base"] = n_base
        rec[f"{pre}_beats_touched"] = int(flags.sum()) if flags.size else 0

        # ---- excluded: drop any beat containing an imputed sample
        keep = [t for t, fl in zip(tails, flags) if not fl]
        f_exc, n_exc = accel_fraction(keep) if len(keep) >= 3 else (np.nan, 0)
        rec[f"{pre}_accel_excl"] = f_exc
        rec[f"{pre}_nbeats_excl"] = n_exc

        # ---- repaired: interpolate short gaps, drop beats with long gaps
        rep = repair(raw.astype(float), mask)
        if rep is None:
            rec[f"{pre}_accel_rep"] = np.nan
            rec[f"{pre}_nbeats_rep"] = 0
        else:
            tr, _ = tails_with_mask(rep, np.zeros_like(mask))
            f_rep, n_rep = accel_fraction(tr)
            rec[f"{pre}_accel_rep"] = f_rep
            rec[f"{pre}_nbeats_rep"] = n_rep
    return rec


# --------------------------------------------------------------------------
def report(df):
    ok = df[df.ok == True]
    print("\n" + "=" * 76)
    print("IMPUTATION SENSITIVITY")
    print("=" * 76)
    print(f"analysable recordings: {len(ok)}\n")

    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        mf = pd.to_numeric(ok[f"{pre}_miss_frac"], errors="coerce")
        touched = pd.to_numeric(ok[f"{pre}_beats_touched"], errors="coerce")
        print(f"{lab}")
        print(f"   missing samples per window: median {100*mf.median():.3f}% "
              f"[{100*mf.quantile(.25):.3f}, {100*mf.quantile(.75):.3f}]")
        print(f"   recordings with any imputation: "
              f"{100*float((mf > 0).mean()):.1f}%")
        print(f"   analysed beats containing an imputed sample: "
              f"median {touched.median():.0f} of 8")
        print(f"   {'regime':<12} {'accel fraction, median [IQR]':>32} {'n':>7}")
        for key, name in (("base", "baseline (median)"),
                          ("excl", "excluded"),
                          ("rep",  "repaired")):
            s = pd.to_numeric(ok[f"{pre}_accel_{key}"], errors="coerce").dropna()
            if s.empty:
                print(f"   {name:<12} {'n/a':>32} {0:7d}")
                continue
            print(f"   {name:<12} "
                  f"{100*s.median():13.1f}% [{100*s.quantile(.25):.1f}, "
                  f"{100*s.quantile(.75):.1f}]".rjust(33) + f" {len(s):7d}")
        # paired difference where both defined
        a = pd.to_numeric(ok[f"{pre}_accel_base"], errors="coerce")
        b = pd.to_numeric(ok[f"{pre}_accel_excl"], errors="coerce")
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 10:
            d = (b - a)[m]
            print(f"   paired change, excluded minus baseline: "
                  f"{100*d.median():+.2f} points "
                  f"[{100*d.quantile(.25):+.2f}, {100*d.quantile(.75):+.2f}] "
                  f"(n = {int(m.sum())})")
        print()

    print("   If the acceleration fraction is unchanged when imputed beats are")
    print("   excluded, median substitution is not producing the result.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results_imputation.csv")
    a = ap.parse_args()

    trks = pd.read_csv("https://api.vitaldb.net/trks")
    ids = sorted(set(trks[trks.tname == "SNUADC/ART"].caseid) &
                 set(trks[trks.tname == "SNUADC/PLETH"].caseid))[:a.n]
    print(f"cohort: {len(ids)} cases")

    rows, done = [], set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        rows = prev.to_dict("records"); done = set(prev.caseid)
        print(f"resuming: {len(done)} done")
    todo = [c for c in ids if int(c) not in done]

    t0 = time.time()
    for i, cid in enumerate(todo, 1):
        rows.append(process_one(cid))
        if i % 20 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(a.out, index=False)
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(todo)} {rate:.2f}/s "
                  f"ETA {(len(todo)-i)/max(rate,1e-9)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    report(df)
    print(f"\nsaved -> {a.out}")
