"""
beats_window.py — reviewer response R2.9: does the shared-tau assumption over
eight beats hold, and do the results depend on the number of beats?

Same 400 recordings (seed 2026) as segmentation_sensitivity.py. From each
120 s window, up to 16 consecutive diastolic tails are collected with the
published pipeline. Then:
  (a) WINDOW LENGTH. Joint fits with the first N = 4, 8, 12, 16 tails, under
      the five published asymptote constraints: median tau and constraint spread.
  (b) LOCAL STATIONARITY. Two non-overlapping blocks of eight tails (tails 1-8
      and 9-16) fitted separately: ratio of the two tau estimates under the
      physiological and unbounded constraints. Under a constant tau and
      asymptote the ratio is 1 up to estimation noise.
Run in Colab:
    !python beats_window.py --c4 .../results_c4.csv --out .../results_beats.csv
"""
import argparse, os, time
import numpy as np, pandas as pd
from vitaldb_pipeline import load_waveforms, segment_quality, detect_feet, FS
from tau_fix import diastole_start
from tau_multibeat import _collect_tails, tau_joint

WIN_S, SKIP_START_S, SEED = 120.0, 600.0, 2026
PRIORS = [("tight", (0.0, 10.0)), ("physio", (0.0, 25.0)), ("wide", (0.0, 40.0)),
          ("permissive", (-20.0, 60.0)), ("free", (-np.inf, np.inf))]
NS = (4, 8, 12, 16)


def fit_all(tails):
    out = {}
    for name, b in PRIORS:
        out[name] = tau_joint(tails, FS, pinf_bounds=b)[0]
    t = [v for v in out.values() if np.isfinite(v)]
    out["spread"] = max(t) / min(t) if len(t) > 1 and min(t) > 0 else np.nan
    return out


def process_one(cid):
    rec = dict(caseid=int(cid), ok=False)
    try:
        v = load_waveforms(cid, fs=FS)
    except Exception:
        return rec
    if v is None or v.shape[0] < FS * (SKIP_START_S + WIN_S):
        return rec
    mid, half = v.shape[0] // 2, int(FS * WIN_S / 2)
    abp, ppg = v[mid - half:mid + half, 0], v[mid - half:mid + half, 1]
    ok, _ = segment_quality(abp, ppg, FS)
    if not ok:
        return rec
    rec["ok"] = True
    for sig, pre in ((abp, "abp"), (ppg, "ppg")):
        sig = np.nan_to_num(sig, nan=float(np.nanmedian(sig)))
        tails = _collect_tails(sig, detect_feet(sig, FS), diastole_start, FS, n_beats=16)
        rec[f"{pre}_ntails"] = len(tails)
        for n in NS:
            if len(tails) >= n:
                r = fit_all(tails[:n])
                rec.update({f"{pre}_N{n}_{k}": x for k, x in r.items()})
        if len(tails) >= 16:
            a, b = fit_all(tails[:8]), fit_all(tails[8:16])
            for k in ("physio", "free"):
                rec[f"{pre}_block_ratio_{k}"] = b[k] / a[k] if a[k] > 0 else np.nan
    return rec


def q(s, d=3):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return "n/a" if s.empty else f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}, {s.quantile(.75):.{d}f}]"


def report(df):
    ok = df[df.ok == True]
    print("\n" + "=" * 84)
    print(f"NUMBER OF BEATS IN THE JOINT FIT (n = {len(ok)} recordings, seed {SEED})")
    print("=" * 84)
    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        print(f"\n{lab}")
        print(f"{'N':>4} {'n':>5} {'tau physio (s)':>24} {'tau free (s)':>24} {'spread':>22} {'<=1.5':>7}")
        for n in NS:
            c = f"{pre}_N{n}_spread"
            if c not in ok:
                continue
            sp = pd.to_numeric(ok[c], errors="coerce").dropna()
            print(f"{n:>4} {len(sp):>5} {q(ok[f'{pre}_N{n}_physio']):>24} {q(ok[f'{pre}_N{n}_free']):>24} "
                  f"{q(sp, 2):>22} {100*np.mean(sp <= 1.5):6.1f}%")
        for k in ("physio", "free"):
            c = f"{pre}_block_ratio_{k}"
            if c in ok:
                r = pd.to_numeric(ok[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                print(f"   tau(beats 9-16) / tau(beats 1-8), {k:>6}: {q(r)}  within 10% of 1: "
                      f"{100*np.mean((r > .9) & (r < 1.1)):.1f}%  (n = {len(r)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results_beats.csv")
    a = ap.parse_args()
    c4 = pd.read_csv(a.c4)
    pool = sorted(int(c) for c in c4.loc[c4.loaded == True, "caseid"])
    ids = sorted(np.random.default_rng(SEED).choice(pool, size=min(a.n, len(pool)), replace=False).tolist())
    rows, done = [], set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out); rows, done = prev.to_dict("records"), set(prev.caseid)
    todo = [c for c in ids if c not in done]
    t0 = time.time()
    for i, cid in enumerate(todo, 1):
        rows.append(process_one(cid))
        if i % 10 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(a.out, index=False)
            r = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(todo)} {r:.2f}/s ETA {(len(todo)-i)/max(r,1e-9)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    report(df)
