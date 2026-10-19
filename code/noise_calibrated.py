"""
noise_calibrated.py — noise-calibrated adequacy test (reviewer R2.8 / R3.3).

The published rule declines a recording if ANY beat shows accelerating decay.
At clinical SNR (~33 dB) noise alone makes ~20% of beats of a perfect
Windkessel waveform accelerate (threshold_validation.py), so that rule cannot
separate model failure from noise. Here each recording is instead tested
against its own noise expectation: with n analysed beats, k of them
accelerating, and p0 the per-beat rate expected from noise at the recording's
measured SNR, the decay is called non-exponential if the one-sided binomial
probability P(K >= k | n, p0) < 0.05.

No waveform processing; uses results_segmentation.csv (base configuration)
and results_snr.csv. Runs in seconds.
    !python noise_calibrated.py --seg .../results_segmentation.csv --snr .../results_snr.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import binom

REF_SNR = np.array([20, 25, 30, 35, 40, 45, 50], float)
REF_ACC = np.array([68.5, 49.9, 32.7, 10.9, 0.9, 0.0, 0.0]) / 100.0


def beats_and_count(frac, n_max):
    """Recover (n, k) from the stored fraction: largest n <= n_max with frac*n integral."""
    if not np.isfinite(frac):
        return np.nan, np.nan
    for n in range(int(n_max), 0, -1):
        k = frac * n
        if abs(k - round(k)) < 1e-6:
            return n, int(round(k))
    return np.nan, np.nan


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", help="results_segmentation.csv (400-case subset)")
    ap.add_argument("--c4", help="results_c4.csv (full adequacy cohort)")
    ap.add_argument("--snr", required=True)
    a = ap.parse_args()
    if a.c4:
        seg = pd.read_csv(a.c4); seg = seg[seg.loaded == True].rename(columns={
            "abp_accel": "base|abp_accel", "ppg_accel": "base|ppg_accel",
            "abp_nbeats": "base|abp_n_tails", "ppg_nbeats": "base|ppg_n_tails"})
    else:
        seg = pd.read_csv(a.seg); seg = seg[seg.loaded == True]
    snr = pd.read_csv(a.snr); snr = snr[snr.ok == True]
    m = seg.merge(snr, on="caseid")
    flags = {}
    print(f"recordings: {len(m)}")
    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        p0 = np.clip(np.interp(m[f"{pre}_snr"], REF_SNR, REF_ACC), 1e-4, None)
        nk = [beats_and_count(f, nt) for f, nt in
              zip(pd.to_numeric(m[f"base|{pre}_accel"], errors="coerce"),
                  pd.to_numeric(m[f"base|{pre}_n_tails"], errors="coerce"))]
        n = np.array([x[0] for x in nk], float); k = np.array([x[1] for x in nk], float)
        ok = np.isfinite(n) & (n >= 4)
        pval = np.where(ok, binom.sf(k - 1, n, p0), np.nan)
        sig = ok & (pval < 0.05)
        flags[pre] = pd.Series(sig, index=m.index)
        cnt, tot = int(sig.sum()), int(ok.sum())
        lo, hi = wilson(cnt, tot)
        print(f"\n{lab}: evaluable {tot}")
        print(f"   acceleration in excess of noise (binomial p < 0.05): {100*cnt/tot:.1f}% [{lo:.1f}, {hi:.1f}]")
        print(f"   published zero-acceleration rule would decline:      {100*np.mean(k[ok] > 0):.1f}%")
        print(f"   median beats analysed {np.nanmedian(n[ok]):.0f}, median noise-expected per-beat rate {100*np.median(p0[ok]):.1f}%")
    both = flags["abp"] & flags["ppg"]
    either = flags["abp"] | flags["ppg"]
    print(f"\nexcess acceleration in both channels: {100*both.mean():.1f}%   in either: {100*either.mean():.1f}%")
    print("\nBy construction, a recording whose acceleration is pure noise is flagged in")
    print("about 5% of cases; rates far above 5% indicate non-exponential decay beyond noise.")
