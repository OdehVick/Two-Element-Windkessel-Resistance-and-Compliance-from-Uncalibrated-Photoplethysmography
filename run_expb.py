"""
run_expb.py — items 5 and 6: sections 4.10 and 4.11.

Run in Colab:
    !python run_expb.py --outdir /content/drive/MyDrive/ppg_ident

SECTION 4.10 -- R and C against a flow-calibrated reference
-----------------------------------------------------------
With cardiac output available, a calibrated reference follows directly:

    R_ref = MAP / CO          C_ref = tau_ABP / R_ref

The plethysmogram supplies no point estimate of R. By Corollary 1 it
constrains only the product tau = RC, so the admissible set is the ridge

    { (R, tau/R) : R > 0 }

truncated to physiological compliance. We therefore report three things
rather than a spurious point comparison:

  (i)   COVERAGE. Does the flow-calibrated reference fall inside the
        admissible interval implied by tau_PPG? If it does, the ridge is
        correct but uninformative; the interval width quantifies how
        uninformative.

  (ii)  INTERVAL WIDTH. The ratio of upper to lower admissible R. This is
        the honest expression of what PPG delivers: not a number, a range.

  (iii) SINGLE-POINT CONVENTIONS. Published methods resolve the ridge by an
        implicit convention -- fixing C at a population value, or learning
        the split from a training cohort. We implement three such
        conventions and compute their agreement with the reference. The
        theory predicts near-zero ICC: the convention, not the signal,
        determines the answer.

Pre-specified criterion H2: falsified if any convention attains ICC > 0.50.

SECTION 4.11 -- downstream propagation
--------------------------------------
Phenylephrine is a pure alpha-1 agonist acting on peripheral resistance. If a
PPG-derived resistance surrogate were a measurement, it should predict
phenylephrine requirement. If it is a population prior in physiological units,
its contribution should vanish after adjustment for tau and demographics.

Pre-specified criterion H3: falsified if the surrogate retains a significant
independent association after adjustment.
"""

import argparse, os, time
import numpy as np
import pandas as pd

from vitaldb_pipeline import (load_waveforms, segment_quality, detect_feet, FS)
from tau_fix import diastole_start
from tau_multibeat import _collect_tails, tau_joint

WIN_S, SKIP_START_S, N_BEATS, MIN_TAILS = 120.0, 600.0, 8, 4
C_LO, C_HI = 0.4, 4.0          # physiological compliance bounds, mL/mmHg
C_POP = 1.5                    # population compliance convention
H2_ICC_MAX = 0.50

CO_TRACKS = ["EV1000/CO", "Vigileo/CO", "CardioQ/CO"]


# ==========================================================================
def process_one(caseid):
    rec = dict(caseid=int(caseid), ok=False, reason="", device="")
    try:
        v = load_waveforms(caseid,
                           tracks=tuple(["SNUADC/ART", "SNUADC/PLETH"] + CO_TRACKS),
                           fs=FS)
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

    # ---- cardiac output: median over the whole case, first available device
    co = np.nan
    for j, tr in enumerate(CO_TRACKS, start=2):
        if v.shape[1] <= j:
            continue
        col = v[:, j]
        col = col[np.isfinite(col)]
        col = col[(col > 0.5) & (col < 15.0)]
        if col.size > 50:
            co = float(np.median(col))
            rec["device"] = tr
            break
    if not np.isfinite(co):
        rec["reason"] = "no usable CO"
        return rec

    # ---- tau in both channels, unconstrained (the only interpretable prior
    #      for a unitless signal; applied to both for symmetry)
    fa, fp = detect_feet(abp, FS), detect_feet(ppg, FS)
    ta = _collect_tails(abp, fa, diastole_start, FS, n_beats=N_BEATS)
    tp = _collect_tails(ppg, fp, diastole_start, FS, n_beats=N_BEATS)
    if len(ta) < MIN_TAILS or len(tp) < MIN_TAILS:
        rec["reason"] = "insufficient beats"
        return rec

    tau_a, r2a, _, pia, _ = tau_joint(ta, FS)
    tau_p, r2p, _, pip, _ = tau_joint(tp, FS)
    if not (np.isfinite(tau_a) and np.isfinite(tau_p)):
        rec["reason"] = "tau fit failed"
        return rec

    mapp = float(np.mean(abp))
    co_ml_s = co * 1000.0 / 60.0
    R_ref = mapp / co_ml_s
    C_ref = tau_a / R_ref

    # ---- admissible ridge implied by tau_PPG, truncated by physiology
    R_lo = tau_p / C_HI
    R_hi = tau_p / C_LO

    # ---- single-point conventions resolving the ridge
    R_pop = tau_p / C_POP                       # fix C at a population value
    R_mid = np.sqrt(R_lo * R_hi)                # geometric centre of the ridge
    # 'oracle' convention: fix C at the cohort median C_ref (filled later)

    rec.update(ok=True, reason="ok",
               tau_abp=tau_a, tau_ppg=tau_p, r2_abp=r2a, r2_ppg=r2p,
               pinf_abp=pia, pinf_ppg=pip,
               map=mapp, co=co, sbp=float(np.percentile(abp, 99)),
               dbp=float(np.percentile(abp, 1)),
               hr=float(60.0 * FS / np.median(np.diff(fa))),
               R_ref=R_ref, C_ref=C_ref,
               R_lo=R_lo, R_hi=R_hi, R_pop=R_pop, R_mid=R_mid,
               covered=bool(R_lo <= R_ref <= R_hi),
               width=float(R_hi / R_lo))
    return rec


# ==========================================================================
def icc21(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = x.size
    if n < 3:
        return np.nan
    M = np.column_stack([x, y]); k = 2; gm = M.mean()
    msr = k * np.sum((M.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((M.mean(0) - gm) ** 2) / (k - 1)
    mse = np.sum((M - M.mean(1, keepdims=True)
                  - M.mean(0, keepdims=True) + gm) ** 2) / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den else np.nan


def qs(s, d=3):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return "n/a" if s.empty else \
        f"{s.median():.{d}f} [{s.quantile(.25):.{d}f}, {s.quantile(.75):.{d}f}]"


def report_410(ok):
    print("\n" + "=" * 78)
    print("SECTION 4.10 -- R and C against a flow-calibrated reference")
    print("=" * 78)
    print(f"analysable cases: {len(ok)}")
    for dev, d in ok.groupby("device"):
        print(f"   {dev:16s} n={len(d):4d}   CO {qs(d.co,2)} L/min")

    print(f"\nreference  R {qs(ok.R_ref)} mmHg s/mL"
          f"    C {qs(ok.C_ref)} mL/mmHg")
    print(f"tau        ABP {qs(ok.tau_abp)} s    PPG {qs(ok.tau_ppg)} s")

    print("\n(i) COVERAGE of the reference by the tau-implied admissible set")
    print(f"    reference inside admissible interval: "
          f"{100*ok.covered.mean():.1f}% of cases")

    print("\n(ii) WIDTH of the admissible interval")
    print(f"    R_hi / R_lo = {qs(ok.width,1)}  "
          f"(fixed at C_HI/C_LO = {C_HI/C_LO:.1f} by construction)")
    print(f"    admissible R spans {qs(ok.R_lo)} to {qs(ok.R_hi)} mmHg s/mL")
    print("    The plethysmogram alone cannot narrow this range: it is the")
    print("    physiological compliance range, not a measurement.")

    print("\n(iii) SINGLE-POINT CONVENTIONS vs the calibrated reference")
    C_oracle = float(pd.to_numeric(ok.C_ref, errors="coerce").median())
    ok = ok.copy()
    ok["R_oracle"] = ok.tau_ppg / C_oracle
    print(f"{'convention':>34} {'ICC(2,1)':>10} {'Pearson r':>11} "
          f"{'median R':>12}")
    rows = [("fix C at population value 1.5", "R_pop"),
            ("geometric centre of the ridge", "R_mid"),
            (f"fix C at cohort median {C_oracle:.2f}", "R_oracle")]
    verdict = []
    for lab, col in rows:
        x = pd.to_numeric(ok.R_ref, errors="coerce")
        y = pd.to_numeric(ok[col], errors="coerce")
        m = np.isfinite(x) & np.isfinite(y)
        icc = icc21(x[m], y[m])
        r = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 3 else np.nan
        verdict.append(icc)
        print(f"{lab:>34} {icc:10.3f} {r:11.3f} {y.median():12.3f}")

    worst = np.nanmax(verdict)
    print(f"\n   H2: falsified if any ICC > {H2_ICC_MAX}. "
          f"max ICC = {worst:.3f} -> "
          f"{'FALSIFIED' if worst > H2_ICC_MAX else 'SUPPORTED'}")
    print("   Note the oracle convention uses the reference cohort's own")
    print("   median compliance and is therefore optimistic by construction.")
    return ok


def report_411(ok, cases_csv="https://api.vitaldb.net/cases"):
    print("\n" + "=" * 78)
    print("SECTION 4.11 -- downstream propagation (phenylephrine)")
    print("=" * 78)
    try:
        cl = pd.read_csv(cases_csv)
    except Exception as e:
        print(f"could not load clinical table: {e}")
        return
    df = ok.merge(cl, on="caseid", how="left", suffixes=("", "_cl"))
    if "intraop_phe" not in df.columns:
        print("intraop_phe not present")
        return
    df["phe"] = (pd.to_numeric(df.intraop_phe, errors="coerce").fillna(0) > 0).astype(int)
    n1 = int(df.phe.sum())
    print(f"cases with usable data: {len(df)}   phenylephrine exposed: {n1} "
          f"({100*n1/max(len(df),1):.1f}%)")
    if n1 < 25:
        print("too few exposed cases for the adjusted model")
        return

    df["R_surrogate"] = df.tau_ppg / C_POP
    feats = ["age", "sex", "bmi"]
    for f in feats:
        if f not in df.columns:
            df[f] = np.nan
    df["sex"] = (df.sex.astype(str).str.upper().str[0] == "M").astype(float)
    d = df[["phe", "R_surrogate", "tau_ppg"] + feats].apply(
        pd.to_numeric, errors="coerce").dropna()
    print(f"complete cases for modelling: {len(d)}")
    if len(d) < 50 or d.phe.sum() < 20:
        print("insufficient complete cases")
        return

    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels unavailable; install with !pip install statsmodels")
        return

    def fit(cols, label):
        X = sm.add_constant(d[cols].astype(float))
        m = sm.Logit(d.phe.astype(float), X).fit(disp=0)
        b = m.params.get("R_surrogate", np.nan)
        p = m.pvalues.get("R_surrogate", np.nan)
        print(f"   {label:<42} OR {np.exp(b):6.3f}   p {p:8.4f}   "
              f"pseudo-R2 {m.prsquared:.4f}")
        return p

    print("\nlogistic regression, outcome = phenylephrine administered")
    p_un = fit(["R_surrogate"], "unadjusted")
    p_adj = fit(["R_surrogate", "tau_ppg"] + feats,
                "adjusted for tau and demographics")

    print(f"\n   H3: falsified if the surrogate retains an independent")
    print(f"   association after adjustment. adjusted p = {p_adj:.4f} -> "
          f"{'FALSIFIED' if p_adj < 0.05 else 'SUPPORTED'}")
    if np.isfinite(p_un) and np.isfinite(p_adj):
        print(f"   unadjusted p = {p_un:.4f}; the surrogate's apparent effect")
        print("   is attributable to tau and demographics if it vanishes here.")


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "results_expb.csv")

    trks = pd.read_csv("https://api.vitaldb.net/trks")
    art = set(trks[trks.tname == "SNUADC/ART"].caseid)
    ple = set(trks[trks.tname == "SNUADC/PLETH"].caseid)
    co = set()
    for tr in CO_TRACKS:
        co |= set(trks[trks.tname == tr].caseid)
    ids = sorted(art & ple & co)
    if a.n:
        ids = ids[:a.n]
    print(f"EXP-B cohort: {len(ids)} cases")

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
            nok = sum(bool(r.get("ok")) for r in rows)
            print(f"  {i}/{len(todo)} ok={nok} {rate:.2f}/s "
                  f"ETA {(len(todo)-i)/max(rate,1e-9)/60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    ok = df[df.ok == True].reset_index(drop=True)

    print("\nquality-gate rejections:")
    for r, n in df[df.ok != True].reason.value_counts().items():
        print(f"   {str(r):26s} {n:5d}")

    if len(ok) >= 10:
        ok = report_410(ok)
        report_411(ok)
    else:
        print("\ntoo few analysable cases")
    print(f"\nsaved -> {path}")
