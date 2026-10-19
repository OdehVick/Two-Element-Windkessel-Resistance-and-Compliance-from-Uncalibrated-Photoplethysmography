"""
estimator_experiment.py — virtual-cohort experiment of Section 2.12 / 3.6,
re-implemented with an explicit leakage-free protocol (reviewer R3.4).

Protocol
  * 350 virtual subjects, ONE waveform per subject, so the 70/30 holdout split
    is a subject-level split (stratification unnecessary, no subject in both sets).
  * Split made once, before any model fitting, with a fixed seed.
  * No data-dependent preprocessing: features are computed per waveform from that
    waveform alone (no scaling, imputation or selection fitted on pooled data);
    gradient-boosted trees need no feature scaling.
  * Hyperparameters fixed a priori (300 trees, depth 3, learning rate 0.1), no tuning.
  * Linear surrogate and isoline sweep use the model trained on the training set only.
Covariate structure: compliance falls with age; resistance rises with age and BMI;
independent sensor gain log-uniform over a twentyfold range (0.25 to 5).
"""
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr, linregress
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from wk_sim import synth_case
from vitaldb_pipeline import detect_feet
from tau_fix import diastole_start
from tau_multibeat import _collect_tails, tau_joint

import sys
FS, N = 500.0, 350
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
rng = np.random.default_rng(SEED)


def features(d):
    y = d["y"]
    tails = _collect_tails(y, detect_feet(y, FS), diastole_start, FS, n_beats=8)
    tau = tau_joint(tails, FS)[0] if len(tails) >= 3 else np.nan
    feet = detect_feet(y, FS)
    beat = y[feet[1]:feet[2]] if feet.size >= 3 else y
    z = (beat - beat.min()) / np.ptp(beat)            # gain- and offset-normalised pulse
    ptr = (z.max() - z.mean()) / max(z.mean() - z.min(), 1e-9)
    return tau, skew(z), kurtosis(z), ptr


def subject(age, male, bmi, R=None, C=None, gain=None):
    if C is None:
        C = float(np.clip(2.4 - 0.022 * (age - 30) + rng.normal(0, 0.15), 0.6, 2.6))
    if R is None:
        R = float(np.clip(0.75 + 0.007 * (age - 30) + 0.015 * (bmi - 25) + rng.normal(0, 0.06), 0.5, 1.9))
    gain = float(np.exp(rng.uniform(np.log(0.25), np.log(5.0)))) if gain is None else gain
    d = synth_case(R, C, level="L0", gain=gain, offset=float(rng.uniform(-1, 1)),
                   hr=float(rng.uniform(62, 82)), sv=float(rng.uniform(60, 80)),
                   rng=rng, snr_db=40.0, fs=FS, n_beats=20, keep_last=14)
    return R, C, gain, features(d)


# ---- virtual cohort ------------------------------------------------------
rows = []
while len(rows) < N:
    age, male, bmi = rng.uniform(30, 80), int(rng.random() < 0.5), float(np.clip(rng.normal(25, 3.5), 18, 40))
    R, C, g, (tau, sk, ku, ptr) = subject(age, male, bmi)
    if np.isfinite(tau):
        rows.append([tau, sk, ku, ptr, age, male, bmi, R])
A = np.array(rows)
X, yR = A[:, :7], A[:, 7]
names = ["tau", "skewness", "kurtosis", "peak ratio", "age", "sex", "BMI"]
idx = np.random.default_rng(SEED + 1).permutation(N)
tr, te = idx[: int(0.7 * N)], idx[int(0.7 * N):]
assert len(set(tr) & set(te)) == 0
gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=SEED)
gbr.fit(X[tr], yR[tr])
p = gbr.predict(X[te])
print(f"subjects {N}, train {len(tr)}, holdout {len(te)} (one waveform per subject; disjoint by construction)")
print(f"holdout r = {pearsonr(p, yR[te])[0]:.3f}, MAE = {np.mean(np.abs(p - yR[te])):.3f} mmHg s/mL")
imp = dict(zip(names, gbr.feature_importances_))
print("importances:", ", ".join(f"{k} {v:.3f}" for k, v in imp.items()))
sur = LinearRegression().fit(X[te][:, [0, 4, 5, 6]], p)
print(f"linear surrogate (tau, age, sex, BMI) on holdout predictions: R2 = {sur.score(X[te][:, [0, 4, 5, 6]], p):.3f}")

# ---- isoline sweep: tau and covariates fixed, true R varied ---------------
tauT, age, male, bmi = 1.60, 55.0, 1, 26.0
Rs = np.array([0.60, 0.80, 1.00, 1.20, 1.45, 1.70]); trueR, pred, taus = [], [], []
for R in Rs:
    for _ in range(20):
        _, _, _, (tau, sk, ku, ptr) = subject(age, male, bmi, R=R, C=tauT / R)
        if np.isfinite(tau):
            trueR.append(R); taus.append(tau); pred.append(gbr.predict([[tau, sk, ku, ptr, age, male, bmi]])[0])
trueR, pred, taus = np.array(trueR), np.array(pred), np.array(taus)
lr = linregress(trueR, pred)
print(f"isoline sweep (n = {len(trueR)}): true R 2.8-fold, predicted R range "
      f"{pred.min():.3f}-{pred.max():.3f}; slope {lr.slope:+.4f} [95% CI {lr.slope-1.96*lr.stderr:+.4f}, {lr.slope+1.96*lr.stderr:+.4f}], r = {lr.rvalue:+.3f}")
for R in Rs:
    m = trueR == R
    print(f"   true R {R:.2f}: predicted {np.mean(pred[m]):.3f} (mean of {m.sum()})")

# ---- Table 8 rows ---------------------------------------------------------
print("\nTable 8 rows: true R, true C, recovered tau median, mean predicted R, error %")
for R in Rs:
    m = trueR == R
    mp = np.mean(pred[m])
    print(f"   {R:.2f}  {tauT/R:.2f}  {np.median(taus[m]):.3f}  {mp:.3f}  {100*(mp-R)/R:+.1f}%")
print(f"predicted R varies {pred[trueR==Rs[-1]].mean()/pred[trueR==Rs[0]].mean():.3f}-fold of means; range ratio of means {max(np.mean(pred[trueR==R]) for R in Rs)/min(np.mean(pred[trueR==R]) for R in Rs):.3f}")
