"""
wk_sim.py — Two-element Windkessel forward simulator with a hierarchy of
observation (transduction) models.

Purpose
-------
Generate synthetic arterial pressure P(t) from a WK2 model with known (R, C),
then produce a synthetic PPG observation y(t) under one of three transduction
classes:

    L0  affine            y = k*P + b                  (gain/offset unknown)
    L1  monotone nonlinear y = k*P^gamma + b           (exponent unknown)
    L2  volumetric        y = k*(V(P) - Vref) + b      (V from Langewouters
                                                        pressure-dependent
                                                        compliance)

These correspond to the three rungs of the identifiability hierarchy:
    L0 -> only tau = R*C is identifiable
    L1 -> nothing is identifiable (tau absorbed by the exponent)
    L2 -> circular: the transduction nonlinearity IS the compliance being sought

Model
-----
    C dP/dt + P/R = Q(t)

Q(t) is a periodic cardiac inflow: half-sine ejection over the systolic
fraction of each beat, zero during diastole.

Units
-----
    P   mmHg
    Q   mL/s
    R   mmHg*s/mL   (SVR; physiological range ~0.7-1.7)
    C   mL/mmHg     (total arterial compliance; range ~0.8-2.5)
    tau = R*C in s  (range ~1.0-2.5)
    V   mL
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------
# Cardiac inflow
# --------------------------------------------------------------------------

def make_inflow(sv=70.0, hr=75.0, sys_frac=1.0 / 3.0):
    """Periodic half-sine ejection waveform.

    Parameters
    ----------
    sv : float
        Stroke volume, mL.
    hr : float
        Heart rate, beats/min.
    sys_frac : float
        Fraction of the cardiac period occupied by ejection.

    Returns
    -------
    Q : callable
        Q(t) -> inflow in mL/s.
    T : float
        Cardiac period in s.
    """
    T = 60.0 / hr
    Ts = sys_frac * T
    # integral of Qpeak*sin(pi*t/Ts) over [0, Ts] = Qpeak*2*Ts/pi = SV
    Qpeak = sv * np.pi / (2.0 * Ts)

    def Q(t):
        phase = np.mod(t, T)
        return np.where(phase < Ts, Qpeak * np.sin(np.pi * phase / Ts), 0.0)

    return Q, T


# --------------------------------------------------------------------------
# WK2 forward solve
# --------------------------------------------------------------------------

@dataclass
class WK2Params:
    R: float                 # mmHg*s/mL
    C: float                 # mL/mmHg
    sv: float = 70.0         # mL
    hr: float = 75.0         # bpm
    sys_frac: float = 1.0 / 3.0

    @property
    def tau(self):
        return self.R * self.C


def simulate_pressure(p: WK2Params, n_beats=12, fs=500.0, keep_last=3):
    """Integrate WK2 to periodic steady state; return the last `keep_last` beats.

    Returns
    -------
    t : ndarray   time, s (re-zeroed at start of retained window)
    P : ndarray   pressure, mmHg
    Q : ndarray   inflow, mL/s
    T : float     cardiac period, s
    """
    Qf, T = make_inflow(p.sv, p.hr, p.sys_frac)

    def rhs(t, y):
        return [(Qf(t) - y[0] / p.R) / p.C]

    t_end = n_beats * T
    # initialise near the expected mean pressure to shorten the transient
    P0 = p.sv / (T / p.R)          # MAP = CO * R = (SV/T) * R
    t_eval = np.arange(0.0, t_end, 1.0 / fs)

    sol = solve_ivp(rhs, (0.0, t_end), [P0], t_eval=t_eval,
                    method="LSODA", rtol=1e-8, atol=1e-10, max_step=1.0 / fs)

    t_all, P_all = sol.t, sol.y[0]
    t_start = (n_beats - keep_last) * T
    m = t_all >= t_start
    t = t_all[m] - t_start
    P = P_all[m]
    Q = Qf(t_all[m])
    return t, P, Q, T


# --------------------------------------------------------------------------
# Transduction models  (P -> observed PPG)
# --------------------------------------------------------------------------

def langewouters_volume(P, Vmax=200.0, P0=70.0, P1=45.0):
    """Volume from the Langewouters arctan pressure-area relation.

    V(P) = (Vmax/pi) * (arctan((P - P0)/P1) + pi/2)
    C(P) = dV/dP = (Vmax/(pi*P1)) / (1 + ((P - P0)/P1)^2)

    Compliance is maximal at P = P0 and falls off at high and low pressure.
    """
    return (Vmax / np.pi) * (np.arctan((P - P0) / P1) + np.pi / 2.0)


def langewouters_compliance(P, Vmax=200.0, P0=70.0, P1=45.0):
    return (Vmax / (np.pi * P1)) / (1.0 + ((P - P0) / P1) ** 2)


@dataclass
class Transduction:
    """Observation model mapping pressure to a PPG-like signal."""
    level: str = "L0"                 # 'L0' | 'L1' | 'L2'
    gain: float = 1.0                 # unknown scale k
    offset: float = 0.0               # unknown baseline b
    gamma: float = 1.0                # L1 only: power-law exponent
    lw: dict = field(default_factory=lambda: dict(Vmax=200.0, P0=70.0, P1=45.0))

    def __call__(self, P):
        if self.level == "L0":
            x = P
        elif self.level == "L1":
            # monotone power law on positive pressure
            x = np.power(np.clip(P, 1e-6, None), self.gamma)
        elif self.level == "L2":
            x = langewouters_volume(P, **self.lw)
        else:
            raise ValueError(f"unknown transduction level {self.level!r}")
        return self.gain * x + self.offset


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------

def add_noise(y, snr_db=None, sigma_frac=None, rng=None):
    """Additive Gaussian noise, specified either by SNR (dB) or by a fraction
    of the signal's peak-to-peak amplitude."""
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=float)
    if snr_db is not None:
        ac = y - y.mean()
        p_sig = np.mean(ac ** 2)
        p_noise = p_sig / (10.0 ** (snr_db / 10.0))
        sigma = np.sqrt(p_noise)
    elif sigma_frac is not None:
        sigma = sigma_frac * np.ptp(y)
    else:
        return y.copy()
    return y + rng.normal(0.0, sigma, size=y.shape)


# --------------------------------------------------------------------------
# One-shot synthetic case
# --------------------------------------------------------------------------

def synth_case(R, C, level="L0", gain=None, offset=None, gamma=None,
               sv=70.0, hr=75.0, fs=500.0, snr_db=30.0, rng=None,
               n_beats=12, keep_last=3):
    """Generate one synthetic (pressure, PPG) pair with known ground truth.

    Unknown nuisance parameters (gain, offset, gamma) are drawn at random
    unless supplied, mimicking unmeasured per-patient sensor characteristics.
    """
    rng = np.random.default_rng() if rng is None else rng
    gain = float(rng.uniform(0.2, 5.0)) if gain is None else gain
    offset = float(rng.uniform(-1.0, 1.0)) if offset is None else offset
    if gamma is None:
        gamma = float(rng.uniform(0.6, 1.6)) if level == "L1" else 1.0

    p = WK2Params(R=R, C=C, sv=sv, hr=hr)
    t, P, Q, T = simulate_pressure(p, n_beats=n_beats, fs=fs, keep_last=keep_last)

    g = Transduction(level=level, gain=gain, offset=offset, gamma=gamma)
    y_clean = g(P)
    y = add_noise(y_clean, snr_db=snr_db, rng=rng)

    return dict(t=t, P=P, Q=Q, y=y, y_clean=y_clean, T=T, fs=fs,
                R=R, C=C, tau=R * C, level=level,
                gain=gain, offset=offset, gamma=gamma, hr=hr, sv=sv)


def physiological_grid(n_R=8, n_C=8, R_range=(0.7, 1.7), C_range=(0.8, 2.5)):
    """Grid of physiologically plausible (R, C) pairs."""
    Rs = np.linspace(*R_range, n_R)
    Cs = np.linspace(*C_range, n_C)
    return [(float(r), float(c)) for r in Rs for c in Cs]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = synth_case(1.0, 1.5, level="L0", rng=rng)
    P = d["P"]
    print(f"R={d['R']} C={d['C']} tau={d['tau']:.3f}s")
    print(f"SBP={P.max():.1f}  DBP={P.min():.1f}  MAP={P.mean():.1f} mmHg")
    print(f"gain={d['gain']:.3f} offset={d['offset']:.3f}")
