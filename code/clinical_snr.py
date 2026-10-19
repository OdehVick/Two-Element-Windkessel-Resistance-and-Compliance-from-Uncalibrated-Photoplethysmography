"""
clinical_snr.py — reviewer response R2.8 / R3.3: is accelerating decay in the
clinical recordings explained by measurement noise?

On synthetic Windkessel waveforms the adequacy diagnostic produces accelerating
beats when noise is high (threshold_validation.py): mean per-beat rate 0% at
>= 45 dB, 0.9% at 40 dB, 10.9% at 35 dB, 32.7% at 30 dB, 49.9% at 25 dB and
68.5% at 20 dB. Whether the clinical figure (median 62.5-75%) is physiological
therefore depends on the SNR of the clinical recordings.

For the same 400 recordings as segmentation_sensitivity.py, this script
estimates the SNR of each channel in the 120 s analysis window with the
estimator used for the synthetic sweep (AC power over the white-noise floor
read from the 40-200 Hz band of the power spectrum; recovers the true SNR of
synthetic signals to within 0.1 dB), and compares the observed per-case
accelerating fraction with the fraction expected from noise alone at that SNR.

Run in Colab (needs results_segmentation.csv from the earlier run):
    !python clinical_snr.py --seg /content/drive/MyDrive/ppg_ident/results_segmentation.csv \
        --out /content/drive/MyDrive/ppg_ident/results_snr.csv
"""
import argparse, os, time
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
from vitaldb_pipeline import load_waveforms, segment_quality, FS

WIN_S, SKIP_START_S = 120.0, 600.0
# synthetic reference: SNR (dB) -> mean per-beat accelerating fraction (%)
REF_SNR = np.array([20, 25, 30, 35, 40, 45, 50], float)
REF_ACC = np.array([68.5, 49.9, 32.7, 10.9, 0.9, 0.0, 0.0])


def snr_db(x, fs=FS, band=(40.0, 200.0)):
    x = np.asarray(x, float); x = x - x.mean()
    f, p = welch(x, fs=fs, nperseg=4096)
    m = (f >= band[0]) & (f <= band[1])
    noise_var = np.median(p[m]) * (fs / 2.0)
    sig_var = max(np.var(x) - noise_var, 1e-30)
    return float(10 * np.log10(sig_var / noise_var))


def expected_accel(snr):
    return float(np.interp(snr, REF_SNR, REF_ACC))


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
    abp = np.nan_to_num(abp, nan=float(np.nanmedian(abp)))
    ppg = np.nan_to_num(ppg, nan=float(np.nanmedian(ppg)))
    rec.update(ok=True, abp_snr=snr_db(abp), ppg_snr=snr_db(ppg))
    return rec


def q(s, d=1):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}, {s.quantile(.75):.{d}f}]"


def report(df, seg):
    m = df[df.ok == True].merge(seg, on="caseid")
    print("\n" + "=" * 78)
    print(f"CLINICAL SNR AND NOISE-EXPECTED ACCELERATION (n = {len(m)})")
    print("=" * 78)
    for pre, lab in (("abp", "arterial line"), ("ppg", "plethysmogram")):
        snr = m[f"{pre}_snr"]
        obs = 100 * pd.to_numeric(m[f"base|{pre}_accel"], errors="coerce")
        exp = snr.apply(expected_accel)
        ok = obs.notna()
        rho = spearmanr(snr[ok], obs[ok])
        print(f"\n{lab}")
        print(f"   SNR (dB)                         {q(snr)}")
        for thr in (45, 40, 35, 30):
            print(f"   recordings with SNR >= {thr} dB      {100*np.mean(snr >= thr):5.1f}%")
        print(f"   observed accelerating beats (%)  {q(obs[ok])}")
        print(f"   expected from noise alone (%)    {q(exp[ok])}")
        print(f"   observed minus expected (pts)    {q((obs - exp)[ok])}")
        hi = ok & (snr >= 45)
        if hi.sum() >= 10:
            print(f"   observed in recordings >= 45 dB  {q(obs[hi])}  (n = {hi.sum()}, noise-expected 0%)")
        print(f"   Spearman rho(SNR, observed)      {rho.correlation:+.3f} (p = {rho.pvalue:.1e})")
    print("\nIf accelerating decay were produced by noise, observed would track expected,")
    print("the correlation with SNR would be strongly negative, and recordings with")
    print("SNR >= 45 dB would show few accelerating beats.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", help="results_segmentation.csv (400-case subset)")
    ap.add_argument("--c4", help="results_c4.csv: run on the full adequacy cohort instead")
    ap.add_argument("--out", default="results_snr.csv")
    a = ap.parse_args()
    if a.c4:
        seg = pd.read_csv(a.c4)
        seg = seg[seg.loaded == True].rename(columns={
            "abp_accel": "base|abp_accel", "ppg_accel": "base|ppg_accel"})
        seg = seg[["caseid", "base|abp_accel", "base|ppg_accel"]]
    else:
        seg = pd.read_csv(a.seg)
        seg = seg[seg.loaded == True][["caseid", "base|abp_accel", "base|ppg_accel"]]
    ids = seg.caseid.astype(int).tolist()
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
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    report(df, seg)
