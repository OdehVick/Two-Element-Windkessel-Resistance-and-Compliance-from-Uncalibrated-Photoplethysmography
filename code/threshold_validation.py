"""
threshold_validation.py — reviewer response R2.8.

Empirical justification of the two prespecified thresholds on synthetic data:
  spread threshold 1.5   (tau considered data-determined if max/min across the
                          five asymptote constraints <= 1.5)
  adequacy criterion     (no accelerating beat AND late/early tau ratio within
                          1/1.3 to 1.3)

FALSE REJECTION. Two-element Windkessel waveforms, for which both criteria
should pass, under additive noise (SNR 15-45 dB), Savitzky-Golay window
(30, 40, 60 ms) and post-notch exclusion (15, 25, 35%).

DETECTION. Waveforms that violate the mono-exponential diastolic model:
  reflection   P(t) + r P(t - 0.15 s), r = 0.1, 0.2, 0.3: a delayed copy of
               the pressure wave, so a reflected systolic wave arrives in diastole
  L2           volumetric (Langewouters) transduction of a Windkessel pressure

Threshold curves: rejection rate for spread thresholds 1.2-3.0 and split
tolerances 1.1-1.6, so the operating point can be read off.

Synthetic only; runs anywhere (no VitalDB access needed).
    python threshold_validation.py
"""
import numpy as np
from wk_sim import synth_case, add_noise, Transduction
from tau_multibeat import tau_joint
import segmentation_sensitivity as ss

FS = 500.0
PRIORS = [(0.0, 10.0), (0.0, 25.0), (0.0, 40.0), (-20.0, 60.0), (-np.inf, np.inf)]
SPREADS = (1.2, 1.5, 2.0, 3.0)
TOLS = (1.1, 1.2, 1.3, 1.5)


def make_signal(kind, rng, snr_db, r=0.0):
    tau = float(rng.uniform(1.0, 2.2)); R = float(rng.uniform(0.8, 1.4)); hr = float(rng.uniform(60, 85))
    d = synth_case(R, tau / R, level="L0", gain=1.0, offset=0.0, hr=hr, rng=rng,
                   snr_db=None, fs=FS, n_beats=20, keep_last=14)
    P = d["P"]
    if kind == "reflection":
        lag = int(0.15 * FS)
        P = P + r * np.r_[np.full(lag, P[0]), P[:-lag]]
        P = P[lag:]
    elif kind == "L2":
        P = Transduction(level="L2", gain=1.0, offset=0.0)(P)
    return add_noise(P, snr_db=snr_db, rng=rng)


def evaluate(sig, skip=0.25, sg_s=0.04):
    cfg = {**ss.BASE, "skip": skip, "sg_s": sg_s}
    tails, _ = ss.collect_tails_p(sig, cfg)
    if len(tails) < 4:
        return None
    te, tl, ratio, _, frac = ss.make_split_test(sg_s)(tails, FS)
    taus = [tau_joint(tails, FS, pinf_bounds=b)[0] for b in PRIORS]
    taus = [t for t in taus if np.isfinite(t)]
    spread = max(taus) / min(taus) if len(taus) > 1 else np.nan
    return dict(ratio=ratio, frac=frac, spread=spread)


def adequate(e, tol):
    return (e["frac"] == 0) and np.isfinite(e["ratio"]) and (1 / tol <= e["ratio"] <= tol)


def run(kind, n, seed, snr_db, r=0.0, skip=0.25, sg_s=0.04):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        e = evaluate(make_signal(kind, rng, snr_db, r), skip, sg_s)
        if e is not None:
            out.append(e)
    return out


def rates(es):
    rej_adq = {t: 100 * np.mean([not adequate(e, t) for e in es]) for t in TOLS}
    rej_spr = {s: 100 * np.mean([not (np.isfinite(e["spread"]) and e["spread"] <= s) for e in es])
               for s in SPREADS}
    acc = 100 * np.mean([e["frac"] > 0 for e in es])
    beat = 100 * np.nanmean([e["frac"] for e in es])
    return rej_adq, rej_spr, acc, beat


if __name__ == "__main__":
    N = 40
    hdr = (f"{'condition':>34} | adequacy rejected (%) at tol " + " ".join(f"{t:>5}" for t in TOLS)
           + " | spread > thr (%) at " + " ".join(f"{s:>4}" for s in SPREADS) + " | cases with any accel %, mean % accel beats")
    print("=" * len(hdr)); print("A. FALSE REJECTION ON WINDKESSEL WAVEFORMS (model holds)"); print("=" * len(hdr)); print(hdr)
    def row(lab, es):
        a, s, acc, beat = rates(es)
        print(f"{lab:>34} | {'':>27}" + " ".join(f"{a[t]:5.1f}" for t in TOLS)
              + " | " + " " * 17 + " ".join(f"{s[x]:4.0f}" for x in SPREADS) + f" | {acc:5.1f}  beats {beat:5.1f}")
    for snr in (50, 45, 40, 35, 30, 25, 20):
        row(f"SNR {snr} dB", run("wk", N, 100 + snr, snr))
    for sg in (0.03, 0.06):
        row(f"SNR 30 dB, SG {int(sg*1000)} ms", run("wk", N, 7, 30, sg_s=sg))
    for sk in (0.15, 0.35):
        row(f"SNR 30 dB, skip {sk}", run("wk", N, 8, 30, skip=sk))
    print("\nB. DETECTION OF NON-MONO-EXPONENTIAL DECAY (model violated), SNR 30 dB")
    print(hdr)
    for r in (0.1, 0.2, 0.3):
        row(f"reflection r = {r}", run("reflection", N, 20 + int(10 * r), 30, r=r))
    row("L2 volumetric transduction", run("L2", N, 31, 30))
