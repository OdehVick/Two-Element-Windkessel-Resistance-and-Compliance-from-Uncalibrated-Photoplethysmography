"""
device_and_demographics.py — the two remaining tables.

1. DEVICE BREAKDOWN for section 3.10. The Method promises EV1000, Vigileo and
   CardioQ separately, because the first two derive cardiac output by
   pulse-contour analysis of the arterial waveform while the third uses
   oesophageal Doppler and is independent of it. The Results currently pool
   them.

2. DEMOGRAPHICS by analysis cohort, for the table the reviewer asked for
   alongside the flow diagram.

Run:
    !python device_and_demographics.py \
        --spread /content/drive/MyDrive/ppg_ident/results_spread.csv \
        --expb   /content/drive/MyDrive/ppg_ident/results_expb.csv \
        --c4     /content/drive/MyDrive/ppg_ident/results_c4.csv
"""

import argparse
import numpy as np
import pandas as pd

TAU_LO, TAU_HI = 0.15, 12.0
C_LO, C_HI, C_POP = 0.4, 4.0, 1.5
PRIORS = ["tight", "physio", "wide", "permissive", "free"]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def q(s, d=3):
    s = num(s).dropna()
    return "n/a" if s.empty else \
        f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}, {s.quantile(.75):.{d}f}]"


def icc21(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = x.size
    if n < 3:
        return np.nan
    M = np.column_stack([x, y]); gm = M.mean()
    msr = 2*np.sum((M.mean(1)-gm)**2)/(n-1)
    msc = n*np.sum((M.mean(0)-gm)**2)
    mse = np.sum((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm)**2)/(n-1)
    den = msr + mse + 2*(msc-mse)/n
    return float((msr-mse)/den) if den else np.nan


def railed(df, pre):
    r = np.zeros(len(df), dtype=bool)
    for p in PRIORS:
        col = f"{pre}_tau_{p}"
        if col not in df:
            continue
        t = num(df[col]).values
        r |= (np.abs(t-TAU_LO) < 1e-3) | (np.abs(t-TAU_HI) < 1e-3)
    return r


# ==========================================================================
def devices(expb_csv):
    print("=" * 78)
    print("1. PER-DEVICE BREAKDOWN  (section 3.10)")
    print("=" * 78)
    df = pd.read_csv(expb_csv)
    ok = df[df.ok == True].copy()
    ok["R_oracle_all"] = num(ok.tau_ppg) / float(num(ok.C_ref).median())

    print(f"{'device':<14} {'n':>5} {'CO (L/min)':>22} "
          f"{'reference R':>22} {'reference C':>22}")
    for dev, g in ok.groupby("device"):
        name = str(dev).split("/")[0]
        print(f"{name:<14} {len(g):5d} {q(g.co,2):>22} "
              f"{q(g.R_ref):>22} {q(g.C_ref):>22}")

    print(f"\n{'device':<14} {'n':>5} "
          f"{'ICC fix C=1.5':>14} {'ICC ridge centre':>18} {'ICC cohort median':>19}")
    for dev, g in ok.groupby("device"):
        name = str(dev).split("/")[0]
        C_or = float(num(g.C_ref).median())
        g = g.copy(); g["R_or"] = num(g.tau_ppg)/C_or
        row = [icc21(num(g.R_ref), num(g[c])) for c in ("R_pop", "R_mid", "R_or")]
        print(f"{name:<14} {len(g):5d} {row[0]:14.3f} {row[1]:18.3f} {row[2]:19.3f}")

    print("\n   EV1000 and Vigileo derive cardiac output by pulse-contour")
    print("   analysis of the arterial waveform; CardioQ uses oesophageal")
    print("   Doppler and is independent of it. Agreement should be compared")
    print("   across devices with that difference in mind.")

    # sensitivity subset, per device
    sub = ok[(~railed(ok, "tau")) if False else
             (num(ok.C_ref).between(C_LO, C_HI))]
    print(f"\n   plausible reference C, by device:")
    for dev, g in ok.groupby("device"):
        name = str(dev).split("/")[0]
        pl = num(g.C_ref).between(C_LO, C_HI)
        print(f"      {name:<12} {int(pl.sum()):4d} of {len(g):4d} "
              f"({100*pl.mean():.1f}%)")
    return ok


# ==========================================================================
def demographics(spread_csv, expb_csv, c4_csv):
    print("\n" + "=" * 78)
    print("2. DEMOGRAPHICS BY ANALYSIS COHORT")
    print("=" * 78)
    cl = pd.read_csv("https://api.vitaldb.net/cases")
    cl["sex_m"] = (cl.sex.astype(str).str.upper().str[0] == "M")

    sp = pd.read_csv(spread_csv)
    ok = sp[sp.ok == True]
    bound_free = ok[~railed(ok, "abp")]
    paired = ok[(~railed(ok, "abp")) & (~railed(ok, "ppg"))]

    eb = pd.read_csv(expb_csv); eb_ok = eb[eb.ok == True]
    c4 = pd.read_csv(c4_csv); c4_ok = c4[c4.loaded == True]

    cohorts = [
        ("EXP-A eligible",        set(sp.caseid)),
        ("Analysed (3.8, 3.9)",   set(ok.caseid)),
        ("Bound-free (3.8.1)",    set(bound_free.caseid)),
        ("Paired (3.12)",         set(paired.caseid)),
        ("Adequacy (3.11)",       set(c4_ok.caseid)),
        ("Reference (3.10)",      set(eb_ok.caseid)),
    ]

    print(f"{'cohort':<22} {'n':>6} {'age (y)':>16} {'female':>9} "
          f"{'BMI':>16} {'ASA III+':>9}")
    rows = []
    for name, ids in cohorts:
        g = cl[cl.caseid.isin(ids)]
        if g.empty:
            continue
        fem = 100*(~g.sex_m).mean()
        asa = num(g.asa)
        asa3 = 100*float((asa >= 3).mean()) if asa.notna().any() else np.nan
        print(f"{name:<22} {len(g):6d} {q(g.age,0):>16} {fem:8.1f}% "
              f"{q(g.bmi,1):>16} {asa3:8.1f}%")
        rows.append(dict(cohort=name, n=len(g)))

    print("\n   Cohorts are nested subsets of EXP-A except the reference")
    print("   cohort, which additionally requires a cardiac output track.")

    # operation type, top categories, for the largest cohort
    g = cl[cl.caseid.isin(set(ok.caseid))]
    if "optype" in g:
        print("\n   operation type, analysed cohort:")
        for k, v in g.optype.value_counts().head(6).items():
            print(f"      {str(k)[:34]:<34} {v:5d} ({100*v/len(g):.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread", required=True)
    ap.add_argument("--expb", required=True)
    ap.add_argument("--c4", required=True)
    a = ap.parse_args()
    devices(a.expb)
    demographics(a.spread, a.expb, a.c4)
