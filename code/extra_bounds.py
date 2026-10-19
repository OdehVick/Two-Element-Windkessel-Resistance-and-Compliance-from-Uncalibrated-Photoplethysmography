"""
extra_bounds.py — reviewer response R1.6 (other asymptote bounds) and
R2.7 / R3.2 (scale-invariant constraints for the unitless plethysmogram).

Same cohort, window, quality gate, beat segmentation and joint estimator as
run_spread.py (the code behind Tables 9 and 10). Only the constraint on the
diastolic asymptote P_inf changes. Two families are fitted:

  1. mmHg family, (0, U) for U = 5 ... 60 mmHg, plus the published
     permissive (-20, 60) and unbounded fits. Contains the published tight,
     physiological and wide bounds, which must reproduce results_spread.csv
     exactly (printed as a consistency check).

  2. Scale-invariant family. For each channel, F0 = median signal value at
     the diastolic feet and AMP = median foot-to-peak pulse amplitude, both in
     the channel's own units. P_inf is bounded to (F0 - f * AMP, F0) for
     f = 1, 2, 4, 8: "the asymptote lies below the diastolic foot by at most
     f pulse amplitudes". Invariant to any gain and offset.

  3. Transported family (plethysmogram only). Each published mmHg bound is
     carried into the plethysmogram's units through the same recording's
     arterial foot and pulse amplitude, P -> F0_ppg + (P - F0_abp)/AMP_abp *
     AMP_ppg. This is the bound the mmHg prior would imply for the PPG if the
     transduction were affine (L0), i.e. the physiologically meaningful
     version of Table 10.

Run in Colab, after copying the project code into the working directory:
    !python extra_bounds.py --spread /content/drive/MyDrive/ppg_ident/results_spread.csv \
        --out /content/drive/MyDrive/ppg_ident/results_extra_bounds.csv
Resumable: rerun the same command after a disconnect. --n 30 for a quick pilot.
"""

import argparse, os, time
import numpy as np
import pandas as pd

from vitaldb_pipeline import load_waveforms, segment_quality, detect_feet, FS
from tau_fix import diastole_start
from tau_multibeat import _collect_tails, tau_joint

WIN_S, SKIP_START_S, N_BEATS, MIN_TAILS = 120.0, 600.0, 8, 4   # as run_spread.py
SPREAD_OK = 1.5

MMHG = [(f"u{u:g}", (0.0, float(u))) for u in (5, 10, 15, 20, 25, 30, 40, 50, 60)]
MMHG += [("permissive", (-20.0, 60.0)), ("free", (-np.inf, np.inf))]
FRAC = (1.0, 2.0, 4.0, 8.0)
TRANSPORT = [("map10", (0.0, 10.0)), ("map25", (0.0, 25.0)), ("map40", (0.0, 40.0)),
             ("mapperm", (-20.0, 60.0))]
PUBLISHED = {"u10": "tight", "u25": "physio", "u40": "wide",
             "permissive": "permissive", "free": "free"}


def channel_scale(sig, feet):
    """Diastolic foot level and pulse amplitude in the channel's own units."""
    f0 = float(np.median(sig[feet]))
    amps = [np.max(sig[a:b]) - sig[a] for a, b in zip(feet[:-1], feet[1:])]
    return f0, float(np.median(amps))


def analyse_channel(sig, fs=FS):
    feet = detect_feet(sig, fs)
    if feet.size < MIN_TAILS + 1:
        return None
    tails = _collect_tails(sig, feet, diastole_start, fs, n_beats=N_BEATS)
    if len(tails) < MIN_TAILS:
        return None
    out = {"n_tails": len(tails)}
    for name, b in MMHG:
        tau, r2, n, pinf, hit = tau_joint(tails, fs, pinf_bounds=b)
        out.update({f"tau_{name}": tau, "r2_" + name: r2,
                    "pinf_" + name: pinf, "hit_" + name: hit})
    f0, amp = channel_scale(sig, feet)
    out.update(foot=f0, amp=amp)
    out["_tails"] = tails
    for f in FRAC:
        name = f"f{f:g}"
        tau, r2, n, pinf, hit = tau_joint(tails, fs, pinf_bounds=(f0 - f * amp, f0))
        out.update({f"tau_{name}": tau, "r2_" + name: r2,
                    "pinf_" + name: pinf, "hit_" + name: hit})
    return out


def process_one(caseid):
    """Identical windowing and quality gate to run_spread.process_one."""
    rec = dict(caseid=int(caseid), ok=False, reason="")
    try:
        v = load_waveforms(caseid, fs=FS)
    except Exception as e:
        rec["reason"] = f"load:{type(e).__name__}"
        return rec
    if v is None or v.shape[0] < FS * (SKIP_START_S + WIN_S):
        rec["reason"] = "too short"
        return rec
    mid, half = v.shape[0] // 2, int(FS * WIN_S / 2)
    abp, ppg = v[mid - half:mid + half, 0], v[mid - half:mid + half, 1]
    ok, reason = segment_quality(abp, ppg, FS)
    if not ok:
        rec["reason"] = reason
        return rec
    abp = np.nan_to_num(abp, nan=float(np.nanmedian(abp)))
    ppg = np.nan_to_num(ppg, nan=float(np.nanmedian(ppg)))
    a, p = analyse_channel(abp), analyse_channel(ppg)
    if a is None or p is None:
        rec["reason"] = "insufficient beats"
        return rec
    tails_p = p.pop("_tails"); a.pop("_tails")
    to_ppg = lambda P: p["foot"] + (P - a["foot"]) / a["amp"] * p["amp"]
    for name, (lo, hi) in TRANSPORT:
        tau, r2, n, pinf, hit = tau_joint(tails_p, FS, pinf_bounds=(to_ppg(lo), to_ppg(hi)))
        p.update({f"tau_{name}": tau, "r2_" + name: r2, "pinf_" + name: pinf, "hit_" + name: hit})
    rec.update(ok=True, reason="ok")
    rec.update({f"abp_{k}": x for k, x in a.items()})
    rec.update({f"ppg_{k}": x for k, x in p.items()})
    return rec


# --------------------------------------------------------------------------
def q(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return "n/a" if s.empty else f"{s.median():.3f} [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]"


def report(df, spread_csv=None):
    ok = df[df.ok == True].copy()
    print("\n" + "=" * 78)
    print(f"EXTRA ASYMPTOTE BOUNDS  (analysable {len(ok)} of {len(df)})")
    print("=" * 78)

    if spread_csv and os.path.exists(spread_csv):
        ref = pd.read_csv(spread_csv)
        ref = ref[ref.ok == True].set_index("caseid").add_prefix("ref_")
        m = ok.set_index("caseid").join(ref, how="inner")
        print(f"\nConsistency with results_spread.csv (n = {len(m)} cases in both)")
        for pre in ("abp", "ppg"):
            for new, old in PUBLISHED.items():
                a = pd.to_numeric(m[f"{pre}_tau_{new}"], errors="coerce")
                b = pd.to_numeric(m[f"ref_{pre}_tau_{old}"], errors="coerce")
                d = (a - b).abs()
                d = d[np.isfinite(d)]
                print(f"   {pre} {old:>11}: max |diff| {d.max():.2e} s, "
                      f"identical (<1e-6) {100*np.mean(d < 1e-6):.1f}%")
        print("   Expected: identical. If not, the pipeline has changed since "
              "the published run.")

    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        print(f"\n{lab}: mmHg family")
        print(f"{'bound':>12} {'tau median [IQR] (s)':>28} {'R2 median [IQR]':>22} {'on bound':>9}")
        for name, (lo, hi) in MMHG:
            hit = ok[f"{pre}_hit_{name}"].astype(float).mean()
            print(f"{f'({lo:g}, {hi:g})':>12} {q(ok[f'{pre}_tau_{name}']):>28} "
                  f"{q(ok[f'{pre}_r2_{name}']):>22} {100*hit:8.1f}%")
        # monotonicity across the nested (0, U) bounds
        cols = [f"{pre}_tau_{n}" for n, _ in MMHG[:9]]
        T = ok[cols].apply(pd.to_numeric, errors="coerce").values
        good = np.all(np.isfinite(T), axis=1)
        mono = np.all(np.diff(T[good], axis=1) <= 0.01 * T[good][:, :-1], axis=1)
        med = np.nanmedian(T, axis=0)
        print(f"   median tau non-increasing as U widens: {bool(np.all(np.diff(med) <= 1e-3 * med[:-1]))}")
        print(f"   cases with tau non-increasing across all nine (0, U) bounds "
              f"(1% tolerance): {100*mono.mean():.1f}% (n = {good.sum()})")

        print(f"{lab}: scale-invariant family, P_inf in (foot - f*AMP, foot)")
        for f in FRAC:
            n = f"f{f:g}"
            hit = ok[f"{pre}_hit_{n}"].astype(float).mean()
            print(f"{f'f = {f:g}':>12} {q(ok[f'{pre}_tau_{n}']):>28} "
                  f"{q(ok[f'{pre}_r2_{n}']):>22} {100*hit:8.1f}%")
        cols = [f"{pre}_tau_f{f:g}" for f in FRAC] + [f"{pre}_tau_free"]
        T = ok[cols].apply(pd.to_numeric, errors="coerce")
        spread = T.max(axis=1) / T.min(axis=1)
        spread = spread[np.isfinite(spread)]
        print(f"   spread across the scale-invariant bounds and the free fit: {q(spread)}  "
              f"<= {SPREAD_OK}: {100*np.mean(spread <= SPREAD_OK):.1f}%  (n = {len(spread)})")

    print("\nplethysmogram: published mmHg bounds transported into PPG units via the arterial channel")
    for name, (lo, hi) in TRANSPORT:
        hit = ok[f"ppg_hit_{name}"].astype(float).mean()
        print(f"{f'({lo:g}, {hi:g})':>12} {q(ok[f'ppg_tau_{name}']):>28} "
              f"{q(ok[f'ppg_r2_{name}']):>22} {100*hit:8.1f}%")
    cols = [f"ppg_tau_{n}" for n, _ in TRANSPORT] + ["ppg_tau_free"]
    T = ok[cols].apply(pd.to_numeric, errors="coerce")
    spread = (T.max(axis=1) / T.min(axis=1)).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"   spread across transported bounds and the free fit: {q(spread)}  "
          f"<= {SPREAD_OK}: {100*np.mean(spread <= SPREAD_OK):.1f}%  (n = {len(spread)})")
    for name, (lo, hi) in TRANSPORT:
        r = (pd.to_numeric(ok[f"ppg_tau_{name}"], errors="coerce") /
             pd.to_numeric(ok[f"abp_tau_u{hi:g}" if lo == 0 else "abp_tau_permissive"], errors="coerce")
             ).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"   PPG/arterial tau ratio under matched bound ({lo:g}, {hi:g}): {q(r)}   "
              f"within 10% of 1: {100*np.mean((r > .9) & (r < 1.1)):.1f}%")

    # channel agreement under the scale-invariant bounds
    print("\nBetween-channel tau ratio (PPG / arterial), scale-invariant bounds")
    for f in FRAC:
        n = f"f{f:g}"
        r = (pd.to_numeric(ok[f"ppg_tau_{n}"], errors="coerce") /
             pd.to_numeric(ok[f"abp_tau_{n}"], errors="coerce")).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"   f = {f:<5g} ratio {q(r)}   within 10% of 1: {100*np.mean((r > .9) & (r < 1.1)):.1f}%")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread", required=True, help="results_spread.csv from the published run")
    ap.add_argument("--out", default="results_extra_bounds.csv")
    ap.add_argument("--n", type=int, default=0, help="pilot: first n cases only")
    a = ap.parse_args()

    ref = pd.read_csv(a.spread)
    ids = sorted(int(c) for c in ref.loc[ref.ok == True, "caseid"])   # the 2866 analysed cases
    if a.n:
        ids = ids[:a.n]
    print(f"cohort: {len(ids)} cases (analysable in results_spread.csv)")

    rows, done = [], set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        rows, done = prev.to_dict("records"), set(prev.caseid)
        print(f"resuming: {len(done)} done")
    todo = [c for c in ids if c not in done]
    t0 = time.time()
    for i, cid in enumerate(todo, 1):
        rows.append(process_one(cid))
        if i % 20 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(a.out, index=False)
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(todo)} {rate:.2f}/s ETA {(len(todo)-i)/max(rate,1e-9)/60:.1f} min",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    report(df, a.spread)
    print(f"\nsaved -> {a.out}")
