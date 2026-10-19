"""
fix_expb.py — corrections to sections 4.10 and 4.11.

Run in Colab (no waveform loading; reads the existing CSV):
    !python fix_expb.py --csv /content/drive/MyDrive/ppg_ident/results_expb.csv

WHAT WAS WRONG
--------------
4.11  The surrogate was defined as R_surrogate = tau_ppg / 1.5, an exact
      linear function of tau_ppg. Entering both in one logistic model makes
      the design matrix singular, so the adjusted p-value was nan and the
      pre-specified criterion fell through to a false PASS. No result.

4.10  The reference C_ref = tau_ABP / R_ref inherits tau_ABP from the
      unconstrained fit, which section 4.8 established is unreliable
      (median 0.440 s; physiological 1.0-2.0 s). The resulting C_ref median
      of 0.411 mL/mmHg is implausible. The reference is therefore not clean
      ground truth and must not be presented as such.

WHAT IS DONE HERE
-----------------
4.11  Reformulated as a nested-model comparison, which is the question
      actually at issue: does the plethysmogram carry information about
      peripheral resistance beyond demographics?

          M0   phe ~ age + sex + bmi
          M1   phe ~ age + sex + bmi + tau_ppg
          M2   phe ~ age + sex + bmi + tau_ppg + R_ref     (positive control)

      tau_ppg is the ONLY identifiable waveform quantity, so it is entered
      directly; no collinear surrogate is constructed. M2 includes the
      flow-calibrated reference resistance as a positive control: if even a
      genuinely calibrated R fails to predict phenylephrine, the outcome is
      simply insensitive and H3 is untestable rather than supported.

      Comparison by likelihood-ratio test and change in AUROC.

4.10  Reported with the reference's own limitations stated, and restricted
      to a sensitivity subset in which neither channel railed and the
      reference C falls in a physiological range.
"""

import argparse
import numpy as np
import pandas as pd

TAU_LO, TAU_HI = 0.15, 12.0
C_LO, C_HI = 0.4, 4.0


def num(s):
    return pd.to_numeric(s, errors="coerce")


def qs(s, d=3):
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
    msr = 2 * np.sum((M.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((M.mean(0) - gm) ** 2)
    mse = np.sum((M - M.mean(1, keepdims=True)
                  - M.mean(0, keepdims=True) + gm) ** 2) / (n - 1)
    den = msr + mse + 2 * (msc - mse) / n
    return float((msr - mse) / den) if den else np.nan


# ==========================================================================
def section_410(ok):
    print("=" * 78)
    print("SECTION 4.10 (corrected) -- R and C against a flow-calibrated reference")
    print("=" * 78)

    ok = ok.copy()
    ok["railed"] = ((np.abs(num(ok.tau_ppg) - TAU_HI) < 1e-3) |
                    (np.abs(num(ok.tau_ppg) - TAU_LO) < 1e-3) |
                    (np.abs(num(ok.tau_abp) - TAU_HI) < 1e-3) |
                    (np.abs(num(ok.tau_abp) - TAU_LO) < 1e-3))
    ok["C_plausible"] = num(ok.C_ref).between(0.4, 4.0)

    print(f"all analysable cases: {len(ok)}")
    print(f"   with a railed tau fit in either channel: "
          f"{int(ok.railed.sum())} ({100*ok.railed.mean():.1f}%)")
    print(f"   with physiologically plausible reference C: "
          f"{int(ok.C_plausible.sum())} ({100*ok.C_plausible.mean():.1f}%)")

    print("\nLIMITATION OF THE REFERENCE. C_ref = tau_ABP / R_ref inherits")
    print("tau_ABP from the unconstrained fit, shown in section 4.8 to be")
    print("unreliable. The reference is therefore not independent ground")
    print("truth; it is a second estimate subject to the same model mismatch.")
    print(f"   reference C: {qs(ok.C_ref)} mL/mmHg "
          f"(physiological range approximately 1.0-2.0)")
    print(f"   reference R: {qs(ok.R_ref)} mmHg s/mL  (plausible)")
    print("   R_ref = MAP/CO depends on no waveform fit and is sound;")
    print("   C_ref does depend on one and should be treated with caution.")

    sub = ok[(~ok.railed) & ok.C_plausible]
    print(f"\nSENSITIVITY SUBSET (no railed fit, plausible reference C): "
          f"n = {len(sub)}")

    for name, d in [("all analysable", ok), ("sensitivity subset", sub)]:
        if len(d) < 10:
            continue
        print(f"\n--- {name} (n = {len(d)}) ---")
        print(f"   coverage of R_ref by the tau-implied set: "
              f"{100*d.covered.mean():.1f}%")
        C_or = float(num(d.C_ref).median())
        d = d.copy()
        d["R_oracle"] = num(d.tau_ppg) / C_or
        print(f"   {'convention':>32} {'ICC':>8} {'Pearson r':>11} {'median R':>10}")
        for lab, col in [("fix C at 1.5", "R_pop"),
                         ("geometric ridge centre", "R_mid"),
                         (f"fix C at cohort median {C_or:.2f}", "R_oracle")]:
            x, y = num(d.R_ref), num(d[col])
            m = np.isfinite(x) & np.isfinite(y)
            r = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 3 else np.nan
            print(f"   {lab:>32} {icc21(x, y):8.3f} {r:11.3f} "
                  f"{y.median():10.3f}")
        meds = [num(d.R_pop).median(), num(d.R_mid).median(),
                num(d.R_oracle).median()]
        print(f"   spread of median R across conventions: "
              f"{max(meds)/min(meds):.2f}-fold, from identical signals")


# ==========================================================================
def section_411(ok, cases_url="https://api.vitaldb.net/cases"):
    print("\n" + "=" * 78)
    print("SECTION 4.11 (corrected) -- downstream propagation")
    print("=" * 78)
    try:
        import statsmodels.api as sm
        from scipy import stats as st
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        print(f"missing dependency: {e}")
        return

    cl = pd.read_csv(cases_url)
    df = ok.merge(cl, on="caseid", how="left", suffixes=("", "_cl"))
    if "intraop_phe" not in df.columns:
        print("intraop_phe absent")
        return

    df["phe"] = (num(df.intraop_phe).fillna(0) > 0).astype(int)
    df["sex_m"] = (df.sex.astype(str).str.upper().str[0] == "M").astype(float)
    df["tau_p"] = num(df.tau_ppg)
    df["R_ref_"] = num(df.R_ref)
    df["age_"] = num(df.age)
    df["bmi_"] = num(df.bmi)

    # exclude railed tau: a bound value is not an estimate
    df = df[np.abs(df.tau_p - TAU_HI) > 1e-3]
    df = df[np.abs(df.tau_p - TAU_LO) > 1e-3]

    d = df[["phe", "tau_p", "R_ref_", "age_", "sex_m", "bmi_"]].dropna()
    print(f"complete cases: {len(d)}   phenylephrine exposed: "
          f"{int(d.phe.sum())} ({100*d.phe.mean():.1f}%)")
    if len(d) < 100 or d.phe.sum() < 25:
        print("insufficient for the nested comparison")
        return

    y = d.phe.astype(float)

    def fit(cols, label):
        X = sm.add_constant(d[cols].astype(float))
        if np.linalg.matrix_rank(X.values) < X.shape[1]:
            print(f"   {label:<44} SINGULAR - collinear predictors")
            return None, np.nan, np.nan
        m = sm.Logit(y, X).fit(disp=0)
        auc = roc_auc_score(y, m.predict(X))
        print(f"   {label:<44} logL {m.llf:9.2f}   AUROC {auc:.4f}")
        return m, m.llf, auc

    print("\nnested logistic models, outcome = phenylephrine administered")
    m0, l0, a0 = fit(["age_", "sex_m", "bmi_"], "M0: demographics")
    m1, l1, a1 = fit(["age_", "sex_m", "bmi_", "tau_p"],
                     "M1: + tau_PPG (only identifiable quantity)")
    m2, l2, a2 = fit(["age_", "sex_m", "bmi_", "tau_p", "R_ref_"],
                     "M2: + R_ref (flow-calibrated, positive control)")

    if m1 is not None and m0 is not None:
        lr = 2 * (l1 - l0)
        p1 = 1 - st.chi2.cdf(lr, 1)
        print(f"\n   M1 vs M0  (does the plethysmogram add anything?)")
        print(f"      LR chi2 = {lr:.3f}, df 1, p = {p1:.4f}   "
              f"delta AUROC = {a1-a0:+.4f}")
    if m2 is not None and m1 is not None:
        lr2 = 2 * (l2 - l1)
        p2 = 1 - st.chi2.cdf(lr2, 1)
        print(f"\n   M2 vs M1  (positive control: does calibrated R add?)")
        print(f"      LR chi2 = {lr2:.3f}, df 1, p = {p2:.4f}   "
              f"delta AUROC = {a2-a1:+.4f}")

        print("\n   INTERPRETATION")
        if p2 >= 0.05:
            print("      The positive control FAILS: even flow-calibrated")
            print("      resistance does not predict phenylephrine use in this")
            print("      cohort. The outcome is therefore insensitive to")
            print("      resistance, and H3 is UNTESTABLE with these data")
            print("      rather than supported. This must be reported as a")
            print("      failed control, not as a confirmatory result.")
        elif p1 < 0.05:
            print("      H3 FALSIFIED: the plethysmogram adds predictive")
            print("      information beyond demographics.")
        else:
            print("      H3 SUPPORTED: calibrated resistance predicts the")
            print("      outcome (positive control passes) while the")
            print("      plethysmogram does not add beyond demographics.")


# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    ok = df[df.ok == True].reset_index(drop=True)
    print(f"loaded {len(df)} cases, {len(ok)} analysable\n")
    section_410(ok)
    section_411(ok)
