"""
analyse_spread.py — items 1-3: analyses computable from results_spread.csv
without any further waveform loading.

Run in Colab:
    !python analyse_spread.py --csv /content/drive/MyDrive/ppg_ident/results_spread.csv

1. MECHANISM TEST
   Section 4.8 attributes the prior dependence to (a) short diastolic windows
   relative to tau and (b) wave-reflection curvature. Both imply testable
   predictions:
       - spread should FALL as the diastolic window lengthens
       - spread should FALL as heart rate falls (longer diastole)
       - the number of observed time constants, dias_s / tau, should be the
         governing variable
   If these do not hold, the stated mechanism is wrong and the text must be
   corrected. This is a genuine falsification test of a claim already in the
   draft, not a confirmatory analysis.

2. TRANSDUCTION EXPONENT
   tau_PPG / tau_ABP estimates 1/gamma per patient (validated in Table 6).
   Its distribution places real physiology on the L0/L1/L2 hierarchy:
       ratio ~ 1     -> affine transduction (L0)
       ratio != 1    -> monotone-nonlinear (L1) or volumetric (L2)
   Computed under every prior, since tau itself is prior-dependent.

3. BOUND-LIMITED CASES
   tau_joint bounds tau to [0.15, 12.0] s, so the maximum possible spread is
   80.0x. Cases attaining that value are bound-limited and must be reported as
   such rather than as measured spreads.
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats

PRIORS = ["tight", "physio", "wide", "permissive", "free"]
TAU_LO, TAU_HI = 0.15, 12.0
SPREAD_MAX = TAU_HI / TAU_LO


def num(s):
    return pd.to_numeric(s, errors="coerce")


def qs(s, d=3):
    s = num(s).dropna()
    if s.empty:
        return "n/a"
    return f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}, {s.quantile(.75):.{d}f}]"


def main(path):
    df = pd.read_csv(path)
    ok = df[df.ok == True].copy()
    print(f"loaded {len(df)} cases, {len(ok)} analysable\n")

    # =====================================================================
    print("=" * 78)
    print("3. BOUND-LIMITED CASES  (reported first: needed to filter 1 and 2)")
    print("=" * 78)
    for pre, lab in [("abp", "arterial line"), ("ppg", "plethysmogram")]:
        n_lo = n_hi = 0
        for p in PRIORS:
            t = num(ok[f"{pre}_tau_{p}"])
            n_lo += int((np.abs(t - TAU_LO) < 1e-3).sum())
            n_hi += int((np.abs(t - TAU_HI) < 1e-3).sum())
        sp = num(ok[f"{pre}_spread"])
        at_max = int((np.abs(sp - SPREAD_MAX) < 0.01).sum())
        print(f"{lab:>16}: fits at tau bounds  low {n_lo:5d}  high {n_hi:5d}  "
              f"(of {len(ok)*len(PRIORS)} fits)")
        print(f"{'':>16}  cases at maximum possible spread ({SPREAD_MAX:.0f}x): "
              f"{at_max} ({100*at_max/len(ok):.1f}%)")
    print("\n   Cases at the maximum are bound-limited: the true spread is")
    print("   unbounded, not 80x. Report as '>= 80x' or exclude and state so.")

    # unbounded-free flag: exclude cases where any fit railed
    for pre in ("abp", "ppg"):
        rail = np.zeros(len(ok), dtype=bool)
        for p in PRIORS:
            t = num(ok[f"{pre}_tau_{p}"]).values
            rail |= (np.abs(t - TAU_LO) < 1e-3) | (np.abs(t - TAU_HI) < 1e-3)
        ok[f"{pre}_railed"] = rail
    print(f"\n   cases with no railed fit:  arterial {(~ok.abp_railed).sum()}"
          f"   plethysmogram {(~ok.ppg_railed).sum()}")

    # =====================================================================
    print("\n" + "=" * 78)
    print("1. MECHANISM TEST -- does short diastole explain the prior dependence?")
    print("=" * 78)

    sub = ok[~ok.abp_railed].copy()
    sub["dias"] = num(sub["abp_dias_s"])
    sub["hr"] = num(sub["hr"])
    sub["spread"] = num(sub["abp_spread"])
    # observed time constants, using the tight-prior tau as a reference scale
    sub["n_tau"] = sub["dias"] / num(sub["abp_tau_tight"])
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["dias", "hr", "spread", "n_tau"])
    print(f"n = {len(sub)} (bound-limited cases excluded)\n")

    print(f"{'predictor':>28} {'Spearman rho':>13} {'p':>12} {'direction':>22}")
    tests = [
        ("diastolic window (s)", "dias", "negative"),
        ("heart rate (min-1)", "hr", "positive"),
        ("observed time constants", "n_tau", "negative"),
    ]
    verdicts = []
    for lab, col, expect in tests:
        r, p = stats.spearmanr(sub[col], sub["spread"])
        got = "negative" if r < 0 else "positive"
        verdicts.append((lab, expect, got, r, p))
        print(f"{lab:>28} {r:13.4f} {p:12.3e} "
              f"{('predicted ' + expect):>22}")

    print("\n   verdict:")
    for lab, expect, got, r, p in verdicts:
        agree = (expect == got) and p < 0.05
        print(f"      {lab:<28} {'SUPPORTS' if agree else 'DOES NOT SUPPORT':>18}"
              f"   (observed {got}, p={p:.1e})")

    print("\n   spread by tertile of diastolic window:")
    sub["dt"] = pd.qcut(sub["dias"], 3, labels=["short", "mid", "long"])
    for g, d in sub.groupby("dt", observed=True):
        print(f"      {str(g):>6}  window {qs(d.dias)} s   "
              f"spread {qs(d.spread, 2)}   n={len(d)}")

    print("\n   spread by tertile of observed time constants (dias/tau):")
    sub["nt"] = pd.qcut(sub["n_tau"], 3, labels=["few", "mid", "many"])
    for g, d in sub.groupby("nt", observed=True):
        print(f"      {str(g):>6}  dias/tau {qs(d.n_tau, 3)}   "
              f"spread {qs(d.spread, 2)}   n={len(d)}")

    # how much variance does window length actually explain?
    r_nt, _ = stats.spearmanr(sub["n_tau"], sub["spread"])
    print(f"\n   Spearman rho^2 for dias/tau vs spread: {r_nt**2:.4f}")
    print("   -> window length explains only this fraction of rank variance;")
    print("      the remainder is consistent with model mismatch that short")
    print("      windows aggravate but do not by themselves cause.")

    # =====================================================================
    print("\n" + "=" * 78)
    print("2. TRANSDUCTION EXPONENT  tau_PPG / tau_ABP  ( = 1/gamma )")
    print("=" * 78)
    print(f"{'prior':>12} {'n':>6} {'ratio median [IQR]':>28} "
          f"{'gamma':>8} {'in [0.9,1.1]':>13}")
    for p in PRIORS:
        a = num(ok[f"abp_tau_{p}"])
        b = num(ok[f"ppg_tau_{p}"])
        keep = (~ok.abp_railed) & (~ok.ppg_railed) & np.isfinite(a) & np.isfinite(b) & (a > 0)
        r = (b[keep] / a[keep]).replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty:
            continue
        frac = float(np.mean((r > 0.9) & (r < 1.1)))
        print(f"{p:>12} {len(r):6d} {qs(r, 3):>28} "
              f"{1/r.median():8.3f} {100*frac:12.1f}%")

    print("\n   L0 (affine transduction) predicts ratio = 1 for every patient.")
    print("   Dispersion far exceeding measurement error indicates that the")
    print("   affine assumption underlying all published PPG-Windkessel")
    print("   methods does not hold at the level of the individual.")

    # dispersion vs. what L0 would predict
    a = num(ok["abp_tau_free"]); b = num(ok["ppg_tau_free"])
    keep = (~ok.abp_railed) & (~ok.ppg_railed) & np.isfinite(a) & np.isfinite(b) & (a > 0)
    r = (b[keep] / a[keep]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) > 10:
        print(f"\n   unconstrained fit: ratio spans "
              f"{r.quantile(.05):.3f} - {r.quantile(.95):.3f} (5th-95th pct)")
        print(f"   implied gamma spans {1/r.quantile(.95):.3f} - "
              f"{1/r.quantile(.05):.3f}")

    # =====================================================================
    print("\n" + "=" * 78)
    print("SUMMARY LINES FOR THE MANUSCRIPT")
    print("=" * 78)
    sp_a = num(ok.abp_spread); sp_p = num(ok.ppg_spread)
    print(f"   spread ABP {qs(sp_a,2)}   PPG {qs(sp_p,2)}")
    print(f"   bound-limited: ABP {100*np.mean(np.abs(sp_a-SPREAD_MAX)<0.01):.1f}%"
          f"   PPG {100*np.mean(np.abs(sp_p-SPREAD_MAX)<0.01):.1f}%")
    for pre, lab in [("abp", "arterial"), ("ppg", "plethysmogram")]:
        r2 = num(ok[f"{pre}_r2_free"]); pl = ok[f"{pre}_free_plausible"].astype(bool)
        print(f"   {lab}: r2>0.95 & implausible {100*((r2>0.95)&~pl).mean():.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    main(a.csv)
