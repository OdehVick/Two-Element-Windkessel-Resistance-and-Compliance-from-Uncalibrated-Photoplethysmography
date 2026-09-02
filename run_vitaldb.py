"""
run_vitaldb.py — Colab driver for EXP-A and EXP-B.

Usage in Colab
--------------
    !pip install -q vitaldb
    # upload vitaldb_pipeline.py and this file, then:
    !python run_vitaldb.py --pilot 20            # pilot first, always
    !python run_vitaldb.py --exp a --n 3459      # full EXP-A
    !python run_vitaldb.py --exp b               # EXP-B (needs CO)

Writes CSVs to ./out/ and prints the tables for sections 4.8-4.11.
Results are checkpointed every 25 cases so a runtime reset costs little.

IMPORTANT: mount Drive first if you want the outputs to survive a reset:
    from google.colab import drive; drive.mount('/content/drive')
    then pass --outdir /content/drive/MyDrive/ppg_ident
"""

import argparse, os, sys, time, json
import numpy as np
import pandas as pd

from vitaldb_pipeline import (find_cohorts, load_waveforms, segment_quality,
                              tau_from_signal, robust_tau, transduction_map,
                              CO_TRACKS, FS)

# --------------------------------------------------------------------------
# Pre-specified thresholds (see Methodology 3.8). Do not change after seeing
# results; if a pilot forces a change, record it and re-run the pilot.
# --------------------------------------------------------------------------
WIN_S        = 120.0     # analysis window length
SKIP_START_S = 600.0     # ignore induction period
MIN_R2_BEAT  = 0.95      # per-beat exponential fit quality
MIN_BEATS    = 20        # minimum accepted beats per case
H1_LOA_PCT   = 25.0      # H1 falsified if LoA exceed +/- this % of mean tau
H2_ICC_MAX   = 0.50      # H2 falsified if ICC(R) or ICC(C) exceeds this


# ==========================================================================
def process_one(caseid, want_co=False):
    """Full per-case analysis. Returns a dict (always), with ok=False on
    rejection and a reason string, so the quality gate can be tabulated."""
    rec = dict(caseid=int(caseid), ok=False, reason="", n_beats_abp=0,
               n_beats_ppg=0)
    try:
        tracks = ["SNUADC/ART", "SNUADC/PLETH"]
        co_track = None
        if want_co:
            for tr in CO_TRACKS:
                if tr.endswith("/CO"):
                    tracks.append(tr)
            co_track = True
        v = load_waveforms(caseid, tracks=tuple(tracks), fs=FS)
    except Exception as e:
        rec["reason"] = f"load error: {type(e).__name__}"
        return rec

    if v is None or v.shape[0] < FS * (SKIP_START_S + WIN_S):
        rec["reason"] = "recording too short"
        return rec

    abp_all, ppg_all = v[:, 0], v[:, 1]
    mid = abp_all.shape[0] // 2
    half = int(FS * WIN_S / 2)
    abp = abp_all[mid - half: mid + half]
    ppg = ppg_all[mid - half: mid + half]

    ok, reason = segment_quality(abp, ppg, FS)
    if not ok:
        rec["reason"] = reason
        return rec

    abp = np.nan_to_num(abp, nan=float(np.nanmedian(abp)))
    ppg = np.nan_to_num(ppg, nan=float(np.nanmedian(ppg)))

    ta, _ = tau_from_signal(abp, FS, min_r2=MIN_R2_BEAT)
    tp, _ = tau_from_signal(ppg, FS, min_r2=MIN_R2_BEAT)
    tau_a, sd_a, n_a = robust_tau(ta)
    tau_p, sd_p, n_p = robust_tau(tp)
    rec.update(n_beats_abp=n_a, n_beats_ppg=n_p)

    if n_a < MIN_BEATS or n_p < MIN_BEATS:
        rec["reason"] = "too few accepted beats"
        return rec

    tm = transduction_map(abp, ppg, FS)

    # flow-calibrated reference, when cardiac output is available
    co_med = np.nan
    if want_co and v.shape[1] > 2:
        co = v[:, 2:]
        co = co[np.isfinite(co).any(axis=1)]
        if co.size:
            col = np.nanmedian(co, axis=0)
            good = col[np.isfinite(col) & (col > 0.5) & (col < 15.0)]
            if good.size:
                co_med = float(good[0])          # L/min

    map_mmhg = float(np.mean(abp))
    R_ref = C_ref = np.nan
    if np.isfinite(co_med):
        co_ml_s = co_med * 1000.0 / 60.0         # L/min -> mL/s
        R_ref = map_mmhg / co_ml_s               # mmHg*s/mL
        C_ref = tau_a / R_ref                    # mL/mmHg

    rec.update(ok=True, reason="ok",
               tau_abp=tau_a, tau_abp_sd=sd_a,
               tau_ppg=tau_p, tau_ppg_sd=sd_p,
               tau_ratio=tau_p / tau_a if tau_a else np.nan,
               gamma_hat=tau_a / tau_p if tau_p else np.nan,
               sbp=float(np.percentile(abp, 99)),
               dbp=float(np.percentile(abp, 1)),
               map=map_mmhg,
               co=co_med, R_ref=R_ref, C_ref=C_ref,
               r2_linear=None if tm is None else tm["r2_linear"],
               curvature=None if tm is None else tm["curvature"])
    return rec


# ==========================================================================
def run(caseids, want_co, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"results_{tag}.csv")
    done = set()
    rows = []
    if os.path.exists(path):
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(prev.caseid)
        print(f"resuming: {len(done)} cases already processed")

    todo = [c for c in caseids if int(c) not in done]
    t0 = time.time()
    for i, cid in enumerate(todo, 1):
        rows.append(process_one(cid, want_co=want_co))
        if i % 25 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(path, index=False)
            el = time.time() - t0
            rate = i / el
            eta = (len(todo) - i) / rate / 60 if rate > 0 else float("nan")
            nok = sum(bool(r.get("ok")) for r in rows)
            print(f"  {i}/{len(todo)}  ok={nok}  "
                  f"{rate:.2f} case/s  ETA {eta:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ==========================================================================
def icc21(x, y):
    """ICC(2,1), two-way random, absolute agreement, single measure."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = x.size
    if n < 3:
        return np.nan
    M = np.column_stack([x, y])
    k = 2
    gm = M.mean()
    ms_r = k * np.sum((M.mean(axis=1) - gm) ** 2) / (n - 1)
    ms_c = n * np.sum((M.mean(axis=0) - gm) ** 2) / (k - 1)
    ss_e = np.sum((M - M.mean(axis=1, keepdims=True)
                   - M.mean(axis=0, keepdims=True) + gm) ** 2)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom != 0 else np.nan


def bland_altman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[m], np.asarray(b)[m]
    d = b - a
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return dict(n=int(a.size), bias=bias, sd=sd,
                loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd,
                mean=float(np.mean(np.column_stack([a, b]))))


# ==========================================================================
def report_exp_a(df):
    print("\n" + "=" * 74)
    print("SECTION 4.8 -- time-constant agreement (EXP-A)")
    print("=" * 74)
    tot = len(df)
    ok = df[df.ok == True]
    print(f"attempted {tot}, passed quality gate {len(ok)} "
          f"({100*len(ok)/max(tot,1):.1f}%)\n")
    print("Table: quality-gate rejections")
    rej = df[df.ok != True].reason.value_counts()
    for r, n in rej.items():
        print(f"   {str(r):32s} {n:5d}  ({100*n/max(tot,1):.1f}%)")

    if len(ok) < 3:
        print("\ninsufficient cases for agreement analysis")
        return
    ba = bland_altman(ok.tau_abp, ok.tau_ppg)
    pct = 100 * max(abs(ba["loa_lo"]), abs(ba["loa_hi"])) / ba["mean"]
    print(f"\ntau_ABP  median {ok.tau_abp.median():.3f} s  "
          f"IQR [{ok.tau_abp.quantile(.25):.3f}, {ok.tau_abp.quantile(.75):.3f}]")
    print(f"tau_PPG  median {ok.tau_ppg.median():.3f} s  "
          f"IQR [{ok.tau_ppg.quantile(.25):.3f}, {ok.tau_ppg.quantile(.75):.3f}]")
    print(f"\nBland-Altman (PPG - ABP), n={ba['n']}")
    print(f"   bias {ba['bias']:+.4f} s   SD {ba['sd']:.4f} s")
    print(f"   95% LoA [{ba['loa_lo']:+.4f}, {ba['loa_hi']:+.4f}] s "
          f"= {pct:.1f}% of mean tau")
    print(f"   ICC(2,1) = {icc21(ok.tau_abp, ok.tau_ppg):.3f}")
    print(f"\n   H1 threshold +/-{H1_LOA_PCT}%  ->  "
          f"{'SUPPORTED' if pct <= H1_LOA_PCT else 'FALSIFIED'}")

    print("\n" + "=" * 74)
    print("SECTION 4.9 -- empirical transduction (EXP-A)")
    print("=" * 74)
    for col, lab in [("r2_linear", "linear R^2 of PPG-vs-ABP"),
                     ("curvature", "normalised curvature"),
                     ("tau_ratio", "tau_PPG / tau_ABP  (= 1/gamma)")]:
        s = pd.to_numeric(ok[col], errors="coerce").dropna()
        if s.empty:
            continue
        print(f"{lab:34s} median {s.median():7.4f}  "
              f"IQR [{s.quantile(.25):7.4f}, {s.quantile(.75):7.4f}]  "
              f"n={len(s)}")
    r = pd.to_numeric(ok.tau_ratio, errors="coerce").dropna()
    if len(r):
        frac = float(np.mean((r > 0.9) & (r < 1.1)))
        print(f"\n   fraction with ratio in [0.9, 1.1] (consistent with L0): "
              f"{100*frac:.1f}%")
        print(f"   implied gamma: median {1/r.median():.3f}")


def report_exp_b(df):
    print("\n" + "=" * 74)
    print("SECTION 4.10 -- R and C against a flow-calibrated reference")
    print("=" * 74)
    ok = df[(df.ok == True) & np.isfinite(df.R_ref)]
    print(f"cases with usable CO: {len(ok)}")
    if len(ok) < 10:
        print("insufficient for agreement analysis")
        return
    print(f"\nreference R  median {ok.R_ref.median():.3f} mmHg*s/mL")
    print(f"reference C  median {ok.C_ref.median():.3f} mL/mmHg")
    print(f"reference CO median {ok.co.median():.2f} L/min")
    print("\nNOTE: a PPG-derived R/C requires an arbitrary point on the tau")
    print("ridge. Report the FULL admissible interval implied by tau, and")
    print("the ICC of any single-point convention against the reference.")
    print(f"\n   H2 threshold ICC > {H2_ICC_MAX} falsifies the analysis.")


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["a", "b"], default="a")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--outdir", default="./out")
    a = ap.parse_args()

    exp_a, exp_b, co_avail = find_cohorts()
    ids = sorted(exp_a if a.exp == "a" else exp_b)
    want_co = (a.exp == "b")

    if a.pilot:
        ids = ids[:a.pilot]
        tag = f"pilot_{a.exp}"
        print(f"\nPILOT: {len(ids)} cases\n")
    else:
        if a.n:
            ids = ids[:a.n]
        tag = f"exp{a.exp}"

    df = run(ids, want_co, a.outdir, tag)
    (report_exp_a if a.exp == "a" else report_exp_b)(df)
    print(f"\nsaved -> {os.path.join(a.outdir, f'results_{tag}.csv')}")
