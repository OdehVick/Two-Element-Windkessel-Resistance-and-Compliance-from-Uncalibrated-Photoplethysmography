"""
verify_theory.py — numerical checks of the identifiability hierarchy.

Claim L0 (affine transduction, unknown gain, unknown inflow scale):
    Reparametrising (R, C) -> (alpha*R, C/alpha) leaves tau = RC fixed and
    scales the pressure waveform by exactly alpha. Under an unknown gain the
    two cases are observationally IDENTICAL over the whole waveform, not just
    in diastole. Only tau is identifiable.

Claim L1 (monotone power-law transduction, unknown exponent):
    y = k * P^gamma turns a diastolic decay exp(-t/tau) into exp(-gamma*t/tau).
    Any (tau, gamma) with the same ratio tau/gamma is observationally
    identical. tau is NOT identifiable.

Claim L2 (volumetric transduction with pressure-dependent compliance):
    The transduction nonlinearity is itself governed by C(P). Recovering C
    from the waveform requires knowing the map that C defines -> circular.
"""

import numpy as np
from wk_sim import (WK2Params, simulate_pressure, Transduction,
                    langewouters_compliance, langewouters_volume, synth_case)


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / np.ptp(b))


# --------------------------------------------------------------------------
print("=" * 74)
print("CLAIM L0 : (R, C) -> (alpha R, C/alpha) scales P by exactly alpha")
print("=" * 74)

R0, C0 = 1.0, 1.5
base = WK2Params(R=R0, C=C0)
t0, P0, Q0, T = simulate_pressure(base, fs=500.0)

print(f"{'alpha':>7} {'R':>7} {'C':>7} {'tau':>7} "
      f"{'max|P_a - a*P_0|/ptp':>22} {'corr(P_a, P_0)':>16}")
for alpha in [0.5, 0.75, 1.25, 2.0, 3.0]:
    p = WK2Params(R=alpha * R0, C=C0 / alpha)
    t1, P1, Q1, _ = simulate_pressure(p, fs=500.0)
    n = min(len(P0), len(P1))
    err = rel_err(P1[:n], alpha * P0[:n])
    corr = float(np.corrcoef(P1[:n], P0[:n])[0, 1])
    print(f"{alpha:7.2f} {alpha*R0:7.3f} {C0/alpha:7.3f} "
          f"{alpha*R0*C0/alpha:7.3f} {err:22.3e} {corr:16.10f}")

print("\n-> P_alpha(t) = alpha * P_0(t) to solver tolerance.")
print("   With unknown gain k, set k_alpha = k/alpha: observations coincide")
print("   EXACTLY over the full cardiac cycle. Only tau is identifiable.\n")


# --------------------------------------------------------------------------
print("=" * 74)
print("CLAIM L0b: observed PPG identical across the tau-isoline")
print("=" * 74)

rng = np.random.default_rng(1)
ys = []
print(f"{'R':>7} {'C':>7} {'tau':>7} {'gain k':>10}   (k chosen as 1/alpha)")
for alpha in [0.5, 1.0, 2.0]:
    p = WK2Params(R=alpha * R0, C=C0 / alpha)
    t1, P1, _, _ = simulate_pressure(p, fs=500.0)
    g = Transduction(level="L0", gain=1.0 / alpha, offset=0.0)
    ys.append(g(P1))
    print(f"{alpha*R0:7.3f} {C0/alpha:7.3f} {alpha*R0*C0/alpha:7.3f} "
          f"{1.0/alpha:10.4f}")

n = min(len(y) for y in ys)
for i in range(1, len(ys)):
    print(f"   max|y_{i} - y_0| / ptp = {rel_err(ys[i][:n], ys[0][:n]):.3e}")
print("\n-> Identical PPG from R differing 4-fold and C differing 4-fold.\n")


# --------------------------------------------------------------------------
print("=" * 74)
print("CLAIM L1 : unknown power-law exponent destroys tau identifiability")
print("=" * 74)

# diastolic segment: P = Pd exp(-t/tau); y = k P^gamma = k' exp(-gamma t/tau)
Pd = 80.0
t = np.linspace(0.0, 0.5, 250)

print(f"{'tau (s)':>9} {'gamma':>7} {'tau/gamma':>10} "
      f"{'max|y - y_ref|/ptp':>20}")
ref = None
for tau, gamma in [(1.5, 1.0), (3.0, 2.0), (0.75, 0.5), (2.25, 1.5)]:
    P = Pd * np.exp(-t / tau)
    y = np.power(P, gamma)
    y = (y - y.min()) / np.ptp(y)          # absorb unknown gain and offset
    if ref is None:
        ref = y
        err = 0.0
    else:
        err = rel_err(y, ref)
    print(f"{tau:9.3f} {gamma:7.3f} {tau/gamma:10.3f} {err:20.3e}")

print("\n-> tau spanning 0.75-3.0 s (4x) is indistinguishable once the")
print("   exponent is free. Under an unknown monotone transduction, even")
print("   tau is unidentifiable.\n")


# --------------------------------------------------------------------------
print("=" * 74)
print("CLAIM L2 : the transduction nonlinearity IS the compliance")
print("=" * 74)

P = np.linspace(50.0, 130.0, 9)
Cp = langewouters_compliance(P)
V = langewouters_volume(P)
print(f"{'P (mmHg)':>10} {'C(P) (mL/mmHg)':>16} {'V(P) (mL)':>12}")
for pp, cc, vv in zip(P, Cp, V):
    print(f"{pp:10.1f} {cc:16.4f} {vv:12.2f}")

span = Cp.max() / Cp.min()
print(f"\n   C(P) varies {span:.2f}x across a normal pressure range.")
print("   PPG tracks V, and dV/dP = C(P). Linearising PPG into pressure")
print("   requires C(P); estimating C(P) requires that linearisation.")
print("   -> circular. This is the Manuscript-A circularity in pure form.\n")


# --------------------------------------------------------------------------
print("=" * 74)
print("SANITY : physiological realism of simulated waveforms")
print("=" * 74)
rng = np.random.default_rng(7)
print(f"{'R':>6} {'C':>6} {'tau':>6} {'SBP':>7} {'DBP':>7} {'MAP':>7} {'PP':>7}")
for R, C in [(0.7, 0.8), (1.0, 1.5), (1.3, 2.0), (1.7, 2.5)]:
    d = synth_case(R, C, level="L0", rng=rng, snr_db=None)
    P = d["P"]
    print(f"{R:6.2f} {C:6.2f} {R*C:6.2f} {P.max():7.1f} {P.min():7.1f} "
          f"{P.mean():7.1f} {np.ptp(P):7.1f}")
print()
