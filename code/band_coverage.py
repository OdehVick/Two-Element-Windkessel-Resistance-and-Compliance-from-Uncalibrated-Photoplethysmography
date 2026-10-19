"""
band_coverage.py — reviewer response R2.12: propagate the constraint dependence
of tau into the admissible (R, C) set.

The published estimator returns the ridge {(R, tau/R): tau/C_HI <= R <= tau/C_LO}
for a single tau. Because tau itself varies with the asymptote convention, the
admissible set is a band, tau_min <= RC <= tau_max, truncated to C_LO <= C <= C_HI.
Its width in R factorises into the convention-dependence of tau and the imposed
physiological compliance range:
    R_max / R_min = (tau_max / tau_min) * (C_HI / C_LO).

tau_min, tau_max for the plethysmogram are taken over the bounds converted into
plethysmographic units (results_extra_bounds.csv) and the unbounded fit. No new
waveform processing; runs in seconds.
    !python band_coverage.py --expb .../results_expb.csv --bounds .../results_extra_bounds.csv
"""
import argparse
import numpy as np, pandas as pd

C_LO, C_HI = 0.4, 4.0
COLS = ["ppg_tau_map10", "ppg_tau_map25", "ppg_tau_map40", "ppg_tau_mapperm", "ppg_tau_free"]


def q(s, d=2):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}, {s.quantile(.75):.{d}f}]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expb", required=True)
    ap.add_argument("--bounds", required=True)
    a = ap.parse_args()
    e = pd.read_csv(a.expb); e = e[e.ok == True]
    b = pd.read_csv(a.bounds); b = b[b.ok == True]
    m = e.merge(b[["caseid"] + COLS], on="caseid", how="inner")
    T = m[COLS].apply(pd.to_numeric, errors="coerce")
    tmin, tmax = T.min(axis=1), T.max(axis=1)
    Rref = pd.to_numeric(m.R_ref, errors="coerce")
    tp = pd.to_numeric(m.tau_ppg, errors="coerce")
    ridge = (tp / C_HI <= Rref) & (Rref <= tp / C_LO)
    band = (tmin / C_HI <= Rref) & (Rref <= tmax / C_LO)
    width = (tmax / tmin) * (C_HI / C_LO)
    print(f"cases with a monitor-derived reference and converted bounds: n = {len(m)}")
    print(f"   tau spread (tau_max / tau_min)          {q(tmax / tmin)}")
    print(f"   band width in R (fold)                  {q(width, 1)}")
    print(f"   of which compliance range (fixed)       {C_HI / C_LO:.1f}-fold")
    print(f"   reference inside single-tau ridge       {100 * ridge.mean():.1f}%")
    print(f"   reference inside band                   {100 * band.mean():.1f}%")
    for dev, g in m.groupby("device"):
        i = g.index
        print(f"   {dev:>12}: n = {len(g):4d}  ridge {100 * ridge[i].mean():5.1f}%  band {100 * band[i].mean():5.1f}%")
