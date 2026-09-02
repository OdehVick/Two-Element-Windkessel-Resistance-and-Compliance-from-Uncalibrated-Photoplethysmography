"""
run_c4_validation.py — contribution C4 applied to the full clinical cohort.

Question
--------
If an estimator reports only what the data determine, how much of a real
intraoperative cohort does it decline, and for what reasons?

This is the number a practitioner needs. Sections 4.8-4.10 establish that
published methods report quantities the data do not support; C4 turns that
into a usable rule. Its value is the decline rate and, critically, WHY it
declines: the split-window model-adequacy test is the diagnostic that no
published PPG-Windkessel method applies.

For each case the estimator is run separately on the arterial line and the
plethysmogram, and returns one of:

    ok        model adequate AND tau prior-insensitive (spread <= 1.5x)
    weak      model adequate but tau varies with the assumed asymptote
    unusable  mono-exponential decay rejected, or fit on a bound, or too
              few usable beats

Output
------
    results_c4.csv       per-case status, split ratio, tau, spread
    printed tables for Results section 4.12
"""

import argparse, os, time
import numpy as np
import pandas as pd

from vitaldb_pipeline import load_waveforms, segment_quality, detect_feet, FS
from tau_fix import diastole_start
from tau_multibeat import _collect_tails
from identifiable_estimator import analyse, split_window_test, SPLIT_TOL

WIN_S, SKIP_START_S, N_BEATS = 120.0, 600.0, 8


def process_one(caseid):
    rec = dict(caseid=int(caseid), loaded=False, reason="")
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
    abp = v[mid - half:mid + half, 0]
    ppg = v[mid - half:mid + half, 1]
    ok, reason = segment_quality(abp, ppg, FS)
    if not ok:
        rec["reason"] = reason
        return rec

    abp = np.nan_to_num(abp, nan=float(np.nanmedian(abp)))
    ppg = np.nan_to_num(ppg, nan=float(np.nanmedian(ppg)))
    rec["loaded"] = True

    for sig, pre in ((abp, "abp"), (ppg, "ppg")):
        tails = _collect_tails(sig, detect_feet(sig, FS), diastole_start,
                               FS, n_beats=N_BEATS)
        r = analyse(tails, FS)
        rec[f"{pre}_status"] = r["status"]
        rec[f"{pre}_reason"] = r["reason"][:60]
        rec[f"{pre}_adequate"] = r["model_adequate"]
        rec[f"{pre}_split"] = r["split_ratio"]
        rec[f"{pre}_accel"] = r.get("frac_accelerating", np.nan)
        rec[f"{pre}_tau"] = r["tau"]
        rec[f"{pre}_spread"] = r["spread"]
        rec[f"{pre}_nbeats"] = r["n_beats"]
    return rec


def report(df):
    tot = len(df)
    ld = df[df.loaded == True]
    print("\n" + "=" * 76)
    print("SECTION 4.12 -- an estimator that reports only identifiable quantities")
    print("=" * 76)
    print(f"cases attempted {tot}; passed signal quality {len(ld)} "
          f"({100*len(ld)/max(tot,1):.1f}%)\n")

    for pre, lab in [("abp", "arterial line"), ("ppg", "plethysmogram")]:
        st = ld[f"{pre}_status"].value_counts()
        n = len(ld)
        print(f"{lab}")
        for k in ("ok", "weak", "unusable"):
            c = int(st.get(k, 0))
            print(f"   {k:>10} {c:6d}  ({100*c/max(n,1):5.1f}%)")
        dec = 100 * (1 - st.get("ok", 0) / max(n, 1))
        print(f"   {'declined':>10} {'':6}  ({dec:5.1f}% not reported as a "
              f"usable estimate)")

        adq = ld[f"{pre}_adequate"].astype(bool)
        print(f"   mono-exponential decay accepted: {100*adq.mean():.1f}%")

        # Accelerating decay, reported as a continuous fraction: no threshold
        acc = pd.to_numeric(ld[f"{pre}_accel"], errors="coerce").dropna()
        if len(acc):
            print(f"   beats with ACCELERATING decay (impossible for any "
                  f"exponential):")
            print(f"      per-case fraction: {100*acc.median():.0f}% "
                  f"[{100*acc.quantile(.25):.0f}%, "
                  f"{100*acc.quantile(.75):.0f}%]")
            print(f"      cases with none: {100*np.mean(acc == 0):.1f}%   "
                  f"cases with all beats: {100*np.mean(acc == 1):.1f}%")

        # split ratio reported only where it is defined, i.e. no accelerating
        # beats; mixing the two categories produced inf-poisoned quantiles
        defined = (pd.to_numeric(ld[f"{pre}_accel"], errors="coerce") == 0)
        sp = pd.to_numeric(ld.loc[defined, f"{pre}_split"],
                           errors="coerce").replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(sp):
            print(f"   split ratio tau_late/tau_early, where defined "
                  f"(n={len(sp)}): {sp.median():.3f} "
                  f"[{sp.quantile(.25):.3f}, {sp.quantile(.75):.3f}]  "
                  f"(tolerance {1/SPLIT_TOL:.2f}-{SPLIT_TOL:.2f})")
        else:
            print("   split ratio: undefined in all cases "
                  "(every case has accelerating beats)")
        print()

    print("reasons for declining, arterial line:")
    r = ld.loc[ld.abp_status != "ok", "abp_reason"].value_counts().head(6)
    for k, v in r.items():
        print(f"   {str(k)[:58]:<58} {v:5d}")
    print("\nreasons for declining, plethysmogram:")
    r = ld.loc[ld.ppg_status != "ok", "ppg_reason"].value_counts().head(6)
    for k, v in r.items():
        print(f"   {str(k)[:58]:<58} {v:5d}")

    both = ld[(ld.abp_status == "ok") & (ld.ppg_status == "ok")]
    print(f"\ncases usable in BOTH channels: {len(both)} "
          f"({100*len(both)/max(len(ld),1):.1f}%)")
    if len(both) > 10:
        a = pd.to_numeric(both.abp_tau, errors="coerce")
        p = pd.to_numeric(both.ppg_tau, errors="coerce")
        m = np.isfinite(a) & np.isfinite(p)
        print(f"   tau ABP {a[m].median():.3f} s   PPG {p[m].median():.3f} s")
        ratio = (p[m] / a[m]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            print(f"   tau_PPG/tau_ABP {ratio.median():.3f} "
                  f"[{ratio.quantile(.25):.3f}, {ratio.quantile(.75):.3f}]  "
                  f"consistent with L0 (within 10% of 1): "
                  f"{100*np.mean((ratio>0.9)&(ratio<1.1)):.1f}%")
        print("\n   These are the cases in which a Windkessel time constant")
        print("   may honestly be reported. Everything else is declined.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "results_c4.csv")

    trks = pd.read_csv("https://api.vitaldb.net/trks")
    ids = sorted(set(trks[trks.tname == "SNUADC/ART"].caseid) &
                 set(trks[trks.tname == "SNUADC/PLETH"].caseid))
    if a.n:
        ids = ids[:a.n]
    print(f"cohort: {len(ids)} cases")

    rows, done = [], set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        rows = prev.to_dict("records"); done = set(prev.caseid)
        print(f"resuming: {len(done)} done")
    todo = [c for c in ids if int(c) not in done]

    t0 = time.time()
    for i, cid in enumerate(todo, 1):
        rows.append(process_one(cid))
        if i % 20 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(path, index=False)
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(todo)} {rate:.2f}/s "
                  f"ETA {(len(todo)-i)/max(rate,1e-9)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    report(df)
    print(f"\nsaved -> {path}")
