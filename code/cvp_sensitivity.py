"""
cvp_sensitivity.py — reviewer response R2.13: does omitting central venous
pressure affect the resistance comparisons?

For the reference cohort of results_expb.csv, the central venous pressure
track (SNUADC/CVP) is loaded where available, averaged over the same 120 s
window, and the reference resistance recomputed as (MAP - CVP) / CO. The
comparisons of §3.10 are then repeated against this corrected reference.
Run in Colab:
    !python cvp_sensitivity.py --expb .../results_expb.csv --out .../results_cvp.csv
"""
import argparse, os, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from vitaldb_pipeline import load_waveforms, FS

WIN_S = 120.0
C_LO, C_HI = 0.4, 4.0


def icc21(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = x.size
    if n < 3:
        return np.nan
    M = np.column_stack([x, y]); gm = M.mean(); k = 2
    msr = k * np.sum((M.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((M.mean(0) - gm) ** 2) / (k - 1)
    mse = np.sum((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2) / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den else np.nan


def process_one(cid):
    rec = dict(caseid=int(cid), cvp_ok=False)
    try:
        v = load_waveforms(cid, tracks=("SNUADC/ART", "SNUADC/CVP"), fs=FS)
    except Exception:
        return rec
    if v is None or v.ndim < 2 or v.shape[1] < 2:
        return rec
    mid, half = v.shape[0] // 2, int(FS * WIN_S / 2)
    abp, cvp = v[mid - half:mid + half, 0], v[mid - half:mid + half, 1]
    good = np.isfinite(cvp) & (cvp > -5) & (cvp < 30)
    if good.mean() < 0.8:
        return rec
    rec.update(cvp_ok=True, cvp=float(np.mean(cvp[good])), map_w=float(np.nanmean(abp)))
    return rec


def q(s, d=3):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}, {s.quantile(.75):.{d}f}]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expb", required=True)
    ap.add_argument("--out", default="results_cvp.csv")
    a = ap.parse_args()
    e = pd.read_csv(a.expb); e = e[e.ok == True]
    ids = e.caseid.astype(int).tolist()
    rows, done = [], set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out); rows, done = prev.to_dict("records"), set(prev.caseid)
    todo = [c for c in ids if c not in done]
    t0 = time.time()
    for i, cid in enumerate(todo, 1):
        rows.append(process_one(cid))
        if i % 20 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(a.out, index=False)
            r = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(todo)} {r:.2f}/s ETA {(len(todo)-i)/max(r,1e-9)/60:.1f} min", flush=True)
    d = e.merge(pd.DataFrame(rows), on="caseid")
    d = d[d.cvp_ok == True].copy()
    co_ml_s = pd.to_numeric(d.co) * 1000.0 / 60.0
    d["R_cvp"] = (pd.to_numeric(d["map"]) - d.cvp) / co_ml_s
    R0, R1 = pd.to_numeric(d.R_ref), d.R_cvp
    print("\n" + "=" * 78)
    print(f"CENTRAL VENOUS PRESSURE SENSITIVITY  (CVP available in {len(d)} of {len(e)} reference cases)")
    print("=" * 78)
    print(f"   CVP (mmHg)                         {q(d.cvp, 1)}")
    print(f"   R = MAP/CO                         {q(R0)}")
    print(f"   R = (MAP - CVP)/CO                 {q(R1)}")
    print(f"   relative reduction                 {q(100 * (R0 - R1) / R0, 1)} %")
    rho = spearmanr(R0, R1)
    print(f"   Spearman rho between the two       {rho.correlation:.3f}")
    print(f"   ICC(2,1) between the two           {icc21(R0, R1):.3f}")
    tp = pd.to_numeric(d.tau_ppg)
    conv = {"fix C at 1.5": tp / 1.5, "ridge centre": np.sqrt((tp / C_HI) * (tp / C_LO)),
            "fix C at cohort median": tp / np.nanmedian(pd.to_numeric(d.tau_abp) / R1)}
    print(f"\n   {'convention':>24} {'ICC vs MAP/CO':>15} {'ICC vs (MAP-CVP)/CO':>21} {'Spearman vs corrected':>23}")
    for k, est in conv.items():
        print(f"   {k:>24} {icc21(est, R0):15.3f} {icc21(est, R1):21.3f} {spearmanr(est, R1).correlation:23.3f}")
    ridge0 = (tp / C_HI <= R0) & (R0 <= tp / C_LO)
    ridge1 = (tp / C_HI <= R1) & (R1 <= tp / C_LO)
    print(f"\n   reference inside single-tau ridge:  {100*ridge0.mean():.1f}% (MAP/CO)  {100*ridge1.mean():.1f}% ((MAP-CVP)/CO)")
