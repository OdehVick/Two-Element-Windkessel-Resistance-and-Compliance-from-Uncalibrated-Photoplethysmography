"""
revised_table15.py — Table 15 recomputed with the noise-calibrated adequacy rule.

Status per recording and channel (same order as identifiable_estimator.analyse):
  declined, excess acceleration : accelerating beats exceed the noise expectation
                                  at the recording's SNR (one-sided binomial p < 0.05)
  declined, slope change        : late/early tau ratio outside 1/2.0-2.0
                                  (tolerance recalibrated: 3.3% false rejection of
                                  Windkessel waveforms at 33 dB, against 40% for 1.3)
  usable with caveat            : adequate, but tau spread across constraints > 1.5
  usable                        : adequate and spread <= 1.5
Inputs are existing outputs only; runs in seconds.
    !python revised_table15.py --c4 .../results_c4.csv --snr .../results_snr_full.csv --spread .../results_spread.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import binom

REF_SNR = np.array([20, 25, 30, 35, 40, 45, 50], float)
REF_ACC = np.array([68.5, 49.9, 32.7, 10.9, 0.9, 0.0, 0.0]) / 100.0
TOL, SPREAD_OK = 2.0, 1.5


def nk(frac, nmax):
    if not (np.isfinite(frac) and np.isfinite(nmax)):
        return np.nan, np.nan
    for n in range(int(nmax), 0, -1):
        if abs(frac * n - round(frac * n)) < 1e-6:
            return n, int(round(frac * n))
    return np.nan, np.nan


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", required=True); ap.add_argument("--snr", required=True); ap.add_argument("--spread", required=True)
    a = ap.parse_args()
    c4 = pd.read_csv(a.c4); c4 = c4[c4.loaded == True]
    snr = pd.read_csv(a.snr); snr = snr[snr.ok == True][["caseid", "abp_snr", "ppg_snr"]]
    sp = pd.read_csv(a.spread)[["caseid", "abp_spread", "ppg_spread"]].rename(
        columns={"abp_spread": "abp_spread_all", "ppg_spread": "ppg_spread_all"})
    m = c4.merge(snr, on="caseid", how="left").merge(sp, on="caseid", how="left")
    status = {}
    print(f"recordings assessed: {len(m)}")
    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        st = []
        for _, r in m.iterrows():
            n, k = nk(r[f"{pre}_accel"], r[f"{pre}_nbeats"])
            if not np.isfinite(n) or n < 3 or not np.isfinite(r[f"{pre}_snr"]):
                st.append("declined, too few beats or no SNR"); continue
            p0 = max(np.interp(r[f"{pre}_snr"], REF_SNR, REF_ACC), 1e-4)
            if binom.sf(k - 1, n, p0) < 0.05:
                st.append("declined, excess acceleration"); continue
            ratio = r[f"{pre}_split"]
            if np.isfinite(ratio) and not (1 / TOL <= ratio <= TOL):
                st.append("declined, slope change"); continue
            s = r[f"{pre}_spread"] if np.isfinite(r[f"{pre}_spread"]) else r[f"{pre}_spread_all"]
            st.append("usable" if np.isfinite(s) and s <= SPREAD_OK else "usable with caveat")
        st = pd.Series(st, index=m.index); status[pre] = st
        N = len(st)
        print(f"\n{lab}")
        for lvl in ("usable", "usable with caveat", "declined, excess acceleration",
                    "declined, slope change", "declined, too few beats or no SNR"):
            c = int((st == lvl).sum()); lo, hi = wilson(c, N) if c else (0.0, 100 * (1 - 0.05 ** (1 / N)))
            print(f"   {lvl:<36} {100*c/N:5.1f}% ({c:4d})  [{lo:.1f}, {hi:.1f}]")
        dec = st.str.startswith("declined"); lo, hi = wilson(int(dec.sum()), N)
        print(f"   {'declined, all reasons':<36} {100*dec.mean():5.1f}% ({int(dec.sum()):4d})  [{lo:.1f}, {hi:.1f}]")
    both = status["abp"].isin(["usable", "usable with caveat"]) & status["ppg"].isin(["usable", "usable with caveat"])
    print(f"\nadequate (usable or caveat) in both channels: {100*both.mean():.1f}% ({int(both.sum())})")
    bu = (status["abp"] == "usable") & (status["ppg"] == "usable")
    print(f"usable in both channels: {100*bu.mean():.1f}% ({int(bu.sum())})")
