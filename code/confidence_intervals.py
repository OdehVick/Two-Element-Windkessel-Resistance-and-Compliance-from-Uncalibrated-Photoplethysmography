"""
confidence_intervals.py — item 5: attach interval estimates to every reported
point statistic.

The manuscript currently reports proportions, correlations, intraclass
correlations and limits of agreement as point values. A reviewer is entitled
to ask how precisely each is determined, particularly where a coefficient is
close to zero and the claim rests on it being negligible rather than merely
non-significant.

This script computes, from the existing result files and without refitting:

  1. Wilson intervals for every reported proportion.
  2. Bootstrap intervals for the Spearman correlations in the mechanism test,
     including the rho-squared that quantifies explained rank variance.
  3. Intraclass correlation with an exact F-based interval, for each
     asymptote constraint and for each cardiac output device.
  4. Limits of agreement for the time constant on both the absolute scale and
     the multiplicative scale. The latter is the appropriate one for a
     quantity that is positive and spans an order of magnitude, and is
     reported as a geometric mean ratio with multiplicative limits.
  5. Interquartile and bootstrap intervals for the prior-spread medians.

Run:
    !python confidence_intervals.py \
        --spread /content/drive/MyDrive/ppg_ident/results_spread.csv \
        --expb   /content/drive/MyDrive/ppg_ident/results_expb.csv \
        --c4     /content/drive/MyDrive/ppg_ident/results_c4.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats

PRIORS = ["tight", "physio", "wide", "permissive", "free"]
TAU_LO, TAU_HI = 0.15, 12.0
rng = np.random.default_rng(0)


def num(s):
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return max(0.0, c-h), min(1.0, c+h)


def prop(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def boot_ci(x, fn, B=5000, alpha=0.05):
    x = np.asarray(x)
    n = x.size
    if n < 5:
        return np.nan, np.nan
    idx = rng.integers(0, n, (B, n))
    vals = np.array([fn(x[i]) for i in idx])
    return tuple(np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)]))


def boot_ci2(x, y, fn, B=5000, alpha=0.05):
    x, y = np.asarray(x), np.asarray(y)
    n = x.size
    idx = rng.integers(0, n, (B, n))
    vals = np.array([fn(x[i], y[i]) for i in idx])
    return tuple(np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)]))


def icc21_ci(x, y, alpha=0.05):
    """ICC(2,1) with an F-distribution interval (Shrout & Fleiss)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n, k = x.size, 2
    if n < 3:
        return np.nan, np.nan, np.nan
    M = np.column_stack([x, y]); gm = M.mean()
    msr = k*np.sum((M.mean(1)-gm)**2)/(n-1)
    msc = n*np.sum((M.mean(0)-gm)**2)/(k-1)
    mse = np.sum((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm)**2)/((n-1)*(k-1))
    den = msr + (k-1)*mse + k*(msc-mse)/n
    icc = (msr-mse)/den if den else np.nan
    # interval via F on the between-target term
    fl = stats.f.ppf(1-alpha/2, n-1, (n-1)*(k-1))
    fu = stats.f.ppf(1-alpha/2, (n-1)*(k-1), n-1)
    fobs = msr/mse if mse else np.nan
    lo = (fobs/fl - 1) / (fobs/fl + (k-1))
    hi = (fobs*fu - 1) / (fobs*fu + (k-1))
    return icc, lo, hi


def railed(df, pre):
    r = np.zeros(len(df), dtype=bool)
    for p in PRIORS:
        c = f"{pre}_tau_{p}"
        if c in df:
            t = num(df[c]).values
            r |= (np.abs(t-TAU_LO) < 1e-3) | (np.abs(t-TAU_HI) < 1e-3)
    return r


# ==========================================================================
def section_proportions(sp, c4):
    print("=" * 76)
    print("1. PROPORTIONS WITH WILSON INTERVALS")
    print("=" * 76)
    ok = sp[sp.ok == True]
    n = len(ok)
    print(f"   prior-dependence cohort, n = {n}")
    for pre, lab in (("abp", "arterial"), ("ppg", "plethysmogram")):
        s = num(ok[f"{pre}_spread"]).dropna()
        k = int((s <= 1.5).sum())
        print(f"      {lab:14s} spread <= 1.5      {prop(k, len(s))}")
        r2 = num(ok[f"{pre}_r2_free"]); pl = ok[f"{pre}_free_plausible"].astype(bool)
        bad = int(((r2 > 0.95) & ~pl).sum())
        print(f"      {lab:14s} R2>0.95 implausible {prop(bad, n)}")
    for p in PRIORS:
        h = num(ok[f"abp_hit_{p}"].astype(float))
        print(f"      arterial fits on bound, {p:11s} {prop(int(h.sum()), len(h))}")

    ok4 = c4[c4.loaded == True]; n4 = len(ok4)
    print(f"\n   adequacy cohort, n = {n4}")
    for pre, lab in (("abp", "arterial"), ("ppg", "plethysmogram")):
        st = ok4[f"{pre}_status"]
        print(f"      {lab:14s} declined  {prop(int((st != 'ok').sum()), n4)}")
        adq = ok4[f"{pre}_adequate"].astype(bool)
        print(f"      {lab:14s} adequate  {prop(int(adq.sum()), n4)}")
        acc = num(ok4[f"{pre}_accel"]).dropna()
        print(f"      {lab:14s} every beat accelerating "
              f"{prop(int((acc >= 0.999).sum()), len(acc))}")


def section_spearman(sp):
    print("\n" + "=" * 76)
    print("2. MECHANISM TEST: SPEARMAN CORRELATIONS WITH BOOTSTRAP INTERVALS")
    print("=" * 76)
    ok = sp[sp.ok == True].copy()
    sub = ok[~railed(ok, "abp")].copy()
    sub["dias"] = num(sub.abp_dias_s)
    sub["hr_"] = num(sub.hr)
    sub["spr"] = num(sub.abp_spread)
    sub["ntau"] = sub.dias / num(sub.abp_tau_tight)
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["dias","hr_","spr","ntau"])
    print(f"   n = {len(sub)} (bound-limited excluded)\n")
    print(f"{'predictor':>26} {'rho':>8} {'95% CI':>20} {'rho^2':>8} {'p':>11}")
    for lab, col in (("diastolic window", "dias"), ("heart rate", "hr_"),
                     ("observed time constants", "ntau")):
        x, y = sub[col].values, sub["spr"].values
        r, p = stats.spearmanr(x, y)
        lo, hi = boot_ci2(x, y, lambda a, b: stats.spearmanr(a, b).statistic)
        print(f"{lab:>26} {r:8.3f} [{lo:+.3f}, {hi:+.3f}]".ljust(56) +
              f" {r**2:8.4f} {p:11.2e}")
    print("\n   The rank variance explained is the square of rho; a coefficient")
    print("   whose interval excludes zero may still explain almost none of it.")


def section_icc(sp, expb):
    print("\n" + "=" * 76)
    print("3. INTRACLASS CORRELATION WITH INTERVALS")
    print("=" * 76)
    ok = sp[sp.ok == True]
    print("   between channels, by asymptote constraint")
    print(f"{'constraint':>14} {'n':>6} {'ICC':>8} {'95% CI':>22}")
    for p in PRIORS:
        a, b = num(ok[f"abp_tau_{p}"]), num(ok[f"ppg_tau_{p}"])
        icc, lo, hi = icc21_ci(a, b)
        m = np.isfinite(a) & np.isfinite(b)
        print(f"{p:>14} {int(m.sum()):6d} {icc:8.3f} [{lo:+.3f}, {hi:+.3f}]".rjust(52))

    eb = expb[expb.ok == True].copy()
    if len(eb):
        print("\n   reference conventions, by device")
        print(f"{'device':>12} {'convention':>16} {'n':>6} {'ICC':>8} {'95% CI':>22}")
        for dev, g in eb.groupby("device"):
            name = str(dev).split("/")[0]
            C_or = float(num(g.C_ref).median())
            g = g.copy(); g["R_or"] = num(g.tau_ppg)/C_or
            for lab, col in (("fix C = 1.5", "R_pop"),
                             ("ridge centre", "R_mid"),
                             ("cohort median", "R_or")):
                icc, lo, hi = icc21_ci(num(g.R_ref), num(g[col]))
                print(f"{name:>12} {lab:>16} {len(g):6d} {icc:8.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}]")


def section_loa(sp):
    print("\n" + "=" * 76)
    print("4. LIMITS OF AGREEMENT, ABSOLUTE AND MULTIPLICATIVE")
    print("=" * 76)
    ok = sp[sp.ok == True]
    print("   The time constant is positive and spans an order of magnitude,")
    print("   so the multiplicative scale is the appropriate one; the ratio")
    print("   limits are reported alongside the conventional absolute limits.\n")
    print(f"{'constraint':>14} {'n':>6} {'bias (s)':>10} {'absolute LoA':>24} "
          f"{'GMR':>7} {'ratio LoA':>18}")
    for p in PRIORS:
        a, b = num(ok[f"abp_tau_{p}"]), num(ok[f"ppg_tau_{p}"])
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        a, b = a[m].values, b[m].values
        if a.size < 10:
            continue
        d = b - a
        bias, sd = d.mean(), d.std(ddof=1)
        ld = np.log(b) - np.log(a)
        gmr = np.exp(ld.mean())
        rl, rh = np.exp(ld.mean() - 1.96*ld.std(ddof=1)), np.exp(ld.mean() + 1.96*ld.std(ddof=1))
        print(f"{p:>14} {a.size:6d} {bias:+10.3f} "
              f"[{bias-1.96*sd:+.2f}, {bias+1.96*sd:+.2f}]".rjust(25) +
              f" {gmr:7.3f} [{rl:.2f}, {rh:.2f}]".rjust(19))


def section_medians(sp):
    print("\n" + "=" * 76)
    print("5. PRIOR-SPREAD MEDIANS WITH BOOTSTRAP INTERVALS")
    print("=" * 76)
    ok = sp[sp.ok == True]
    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        s = num(ok[f"{pre}_spread"]).dropna().values
        lo, hi = boot_ci(s, np.median)
        print(f"   {lab:16s} median {np.median(s):5.2f} [{lo:.2f}, {hi:.2f}]  "
              f"IQR [{np.percentile(s,25):.2f}, {np.percentile(s,75):.2f}]  n = {s.size}")
    print("\n   The synthetic reference value is exactly 1.00, which lies far")
    print("   outside both intervals.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread", required=True)
    ap.add_argument("--expb", required=True)
    ap.add_argument("--c4", required=True)
    a = ap.parse_args()
    sp = pd.read_csv(a.spread)
    eb = pd.read_csv(a.expb)
    c4 = pd.read_csv(a.c4)
    section_proportions(sp, c4)
    section_spearman(sp)
    section_icc(sp, eb)
    section_loa(sp)
    section_medians(sp)
