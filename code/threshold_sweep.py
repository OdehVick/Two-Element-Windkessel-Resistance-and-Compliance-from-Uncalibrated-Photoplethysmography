"""
threshold_sweep.py — item 3: is the near-total decline an artefact of the
zero-failure criterion?

The estimator declares a recording adequate only if NO beat shows accelerating
decay. A reviewer may reasonably ask whether that rule, rather than the data,
produces the 99.2% decline rate. This script answers it directly by
recomputing the pass rate at a range of thresholds, using the per-case
acceleration fractions already stored in results_c4.csv. No refitting.

It also reports Wilson confidence intervals on the headline proportions, and
the fraction of the observed rate that any tolerable threshold could rescue.

Run:
    !python threshold_sweep.py --csv /content/drive/MyDrive/ppg_ident/results_c4.csv
"""

import argparse
import numpy as np
import pandas as pd


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return (max(0.0, c-h), min(1.0, c+h))


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def main(csv):
    df = pd.read_csv(csv)
    ok = df[df.loaded == True].copy()
    n = len(ok)
    print(f"analysable recordings: {n}\n")

    A = pd.to_numeric(ok.abp_accel, errors="coerce")
    P = pd.to_numeric(ok.ppg_accel, errors="coerce")

    # ------------------------------------------------------------------
    print("=" * 74)
    print("1. PASS RATE vs ACCELERATION THRESHOLD")
    print("=" * 74)
    print("   A recording passes if the fraction of its beats showing")
    print("   accelerating decay is at or below the threshold.\n")
    print(f"{'threshold':>10} {'arterial line':>26} {'plethysmogram':>26}")
    for th in [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75]:
        a = A.dropna(); p = P.dropna()
        ka = int((a <= th).sum()); kp = int((p <= th).sum())
        print(f"{th:10.2f} {fmt_ci(ka, len(a)):>26} {fmt_ci(kp, len(p)):>26}")

    print("\n   The criterion used throughout is a threshold of zero. If the")
    print("   pass rate remains low at 0.20 or 0.50, the decline is a property")
    print("   of the data rather than of the rule.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("2. DISTRIBUTION OF THE PER-CASE ACCELERATION FRACTION")
    print("=" * 74)
    for lab, s in (("arterial line", A), ("plethysmogram", P)):
        s = s.dropna()
        print(f"   {lab:16s} median {100*s.median():5.1f}% "
              f"[{100*s.quantile(.25):.1f}, {100*s.quantile(.75):.1f}]  "
              f"n = {len(s)}")
        # where does the mass sit?
        bins = [0, 0.001, 0.10, 0.25, 0.50, 0.75, 0.999, 1.001]
        labs = ["0%", "0-10%", "10-25%", "25-50%", "50-75%", "75-100%", "100%"]
        cnt = pd.cut(s, bins=bins, right=False, labels=labs).value_counts().reindex(labs)
        for l, c in cnt.items():
            bar = "#" * int(40 * c / max(cnt.max(), 1))
            print(f"      {l:>8} {int(c):5d}  {bar}")
        print()

    # ------------------------------------------------------------------
    print("=" * 74)
    print("3. HEADLINE PROPORTIONS WITH WILSON INTERVALS")
    print("=" * 74)
    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        st = ok[f"{pre}_status"]
        for key in ("ok", "weak", "unusable"):
            k = int((st == key).sum())
            print(f"   {lab:16s} {key:9s} {fmt_ci(k, n)}")
        dec = int((st != "ok").sum())
        print(f"   {lab:16s} {'declined':9s} {fmt_ci(dec, n)}")
        adq = ok[f"{pre}_adequate"].astype(bool)
        print(f"   {lab:16s} {'adequate':9s} {fmt_ci(int(adq.sum()), n)}\n")

    # ------------------------------------------------------------------
    print("=" * 74)
    print("4. WHAT A RELAXED CRITERION WOULD RECOVER")
    print("=" * 74)
    for lab, s in (("arterial line", A.dropna()), ("plethysmogram", P.dropna())):
        base = float((s <= 0).mean())
        for th in (0.10, 0.20, 0.50):
            got = float((s <= th).mean())
            print(f"   {lab:16s} threshold {th:.2f}: pass rate "
                  f"{100*base:.1f}% -> {100*got:.1f}%  "
                  f"(+{100*(got-base):.1f} points)")
        print()
    print("   A criterion permitting half the beats to contradict the model")
    print("   would not be defensible, but the comparison shows how little")
    print("   the reported decline depends on the choice.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    main(a.csv)
