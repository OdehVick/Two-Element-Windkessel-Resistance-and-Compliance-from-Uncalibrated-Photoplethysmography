"""
identifiability_hierarchy.py — reviewer response R2.1-R2.4, R3.1.

Full-waveform structural and practical identifiability of the two-element
Windkessel under the three transduction classes, with KNOWN inflow.

Forward model: exact periodic steady state, computed harmonic by harmonic from
the transfer function R / (1 + i w R C) applied to the half-sine inflow of
wk_sim.make_inflow (SV 70 mL, HR 75 min-1, ejection one third of the cycle).

Observation models
    L0  y = k P + b
    L1  y = k P^gamma + b
    L2  y = K [arctan((P - P0)/P1) + pi/2] + b      (Langewouters, K = k Vmax / pi)

For each class: (1) singular values of the sensitivity matrix in log
parameters at many physiological parameter sets, and the angle between the
null vector and the analytically predicted scaling direction; (2) a profile
likelihood in tau at a realistic noise level, giving the width of the 95%
profile interval for tau.
"""
import numpy as np
from scipy.optimize import least_squares

FS, HR, SV, SYS = 500.0, 75.0, 70.0, 1.0 / 3.0
T = 60.0 / HR
N = int(round(FS * T))
t = np.arange(N) / FS
Ts = SYS * T
q = np.where(t < Ts, SV * np.pi / (2 * Ts) * np.sin(np.pi * t / Ts), 0.0)
Qf = np.fft.rfft(q)
w = 2 * np.pi * np.fft.rfftfreq(N, 1 / FS)


def pressure(R, C):
    return np.fft.irfft(Qf * R / (1 + 1j * w * R * C), N)


def obs(th, cls):
    if cls == "L0":
        R, C, k = np.exp(th[:3]); b = th[3]
        return k * pressure(R, C) + b
    if cls == "L1":
        R, C, k, g = np.exp(th[:4]); b = th[4]
        return k * pressure(R, C) ** g + b
    if cls == "L2":
        R, C, K, P0, P1 = np.exp(th[:5]); b = th[5]
        return K * (np.arctan((pressure(R, C) - P0) / P1) + np.pi / 2) + b


# predicted null direction (log-parameters, offset last) from the scaling
#   R -> aR, C -> C/a, and the compensating change in the transduction
def predicted_null(cls, th):
    if cls == "L0":
        v = [1, -1, -1, 0]
    elif cls == "L1":
        v = [1, -1, -np.exp(th[3]), 0, 0]
    else:
        v = [1, -1, 0, 1, 1, 0]
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def jac(th, cls, h=1e-6):
    return np.array([(obs(th + h * e, cls) - obs(th - h * e, cls)) / (2 * h)
                     for e in np.eye(len(th))]).T


def theta(cls, R, C, gamma=1.3):
    if cls == "L0":
        return np.array([np.log(R), np.log(C), 0.0, 0.0])
    if cls == "L1":
        return np.array([np.log(R), np.log(C), 0.0, np.log(gamma), 0.0])
    return np.array([np.log(R), np.log(C), np.log(60.0), np.log(70.0), np.log(45.0), 0.0])


def structural(cls, grid, gammas=(0.7, 1.3)):
    rows = []
    for R, C in grid:
        for g in (gammas if cls == "L1" else (None,)):
            th = theta(cls, R, C, g or 1.3)
            J = jac(th, cls)
            Jn = J / np.linalg.norm(J, axis=0)          # column-normalised
            U, s, Vt = np.linalg.svd(Jn, full_matrices=False)
            # null vector back in un-normalised log coordinates
            v = Vt[-1] / np.linalg.norm(J, axis=0)
            v /= np.linalg.norm(v)
            ang = np.degrees(np.arccos(min(1.0, abs(v @ predicted_null(cls, th)))))
            rows.append((s[-1] / s[0], s[-2] / s[0], ang))
    return np.array(rows)


def profile_tau(cls, R, C, gamma=1.3, noise_frac=0.01, n_beats=8,
                rel=(0.8, 0.9, 0.95, 1.05, 1.1, 1.25)):
    """Profile SSE over tau (C = tau/R with R fixed: the scaling direction makes
    fixing R free of loss). Returns dchi2 per relative tau and the 95% interval."""
    th0 = theta(cls, R, C, gamma)
    y0 = obs(th0, cls)
    sigma = noise_frac * np.ptp(y0)
    tau0 = R * C
    out = []
    for r in rel:
        tau = r * tau0
        def res(z):
            th = th0.copy()
            th[1] = np.log(tau / R)
            th[2:] = z
            return (obs(th, cls) - y0) / sigma
        best = None
        for jit in (0.0, 0.3, -0.3):
            z0 = th0[2:].copy(); z0[:-1] += jit * (np.arange(len(z0) - 1) % 2 * 2 - 1) * 0.5
            try:
                fit = least_squares(res, z0, x_scale="jac", max_nfev=4000)
            except Exception:
                continue
            if best is None or fit.cost < best.cost:
                best = fit
        out.append((r, 2 * best.cost * n_beats))      # chi2 over n_beats identical cycles
    return out



# ---- diastolic decay only: 0.36 s window (clinical median), Q = 0 ----
def diastolic_report():
    FS=500.; td=np.arange(int(0.36*FS))/FS        # median clinical diastolic window
    def obs_d(th,cls):
        if cls=='L0': A,tau,k=np.exp(th[:3]); b=th[3]; return k*A*np.exp(-td/tau)+b
        if cls=='L1': A,tau,k,g=np.exp(th[:4]); b=th[4]; return k*(A*np.exp(-td/tau))**g+b
        A,tau,K,P0,P1=np.exp(th[:5]); b=th[5]; return K*(np.arctan((A*np.exp(-td/tau)-P0)/P1)+np.pi/2)+b
    def sv_d(th,cls):
        J=np.array([(obs_d(th+1e-6*e,cls)-obs_d(th-1e-6*e,cls))/2e-6 for e in np.eye(len(th))]).T
        J/=np.linalg.norm(J,axis=0); s=np.linalg.svd(J,compute_uv=False); return s/s[0]
    for cls,th in [('L0',[np.log(90),np.log(1.5),0,0]),('L1',[np.log(90),np.log(1.5),0,np.log(1.3),0]),
                   ('L2',[np.log(90),np.log(1.5),np.log(60),np.log(70),np.log(45),0])]:
        print(cls,'diastole-only relative singular values',np.array2string(sv_d(np.array(th,float),cls),precision=1))
    # profile tau for L2 diastole-only (8 beats, 1% noise)
    th0=np.array([np.log(90),np.log(1.5),np.log(60),np.log(70),np.log(45),0.]); y0=obs_d(th0,'L2'); sig=0.01*np.ptp(y0)
    for r in (0.5,0.8,1.25,2.0):
        def res(z):
            th=th0.copy(); th[1]=np.log(1.5*r); th[[0,2,4,5]]=z; return (obs_d(th,'L2')-y0)/sig   # P0 fixed: scaling direction
        f=min((least_squares(res,th0[[0,2,4,5]]+d,x_scale='jac') for d in (0,0.2,-0.2)),key=lambda f:f.cost)
        print(f'L2 diastole-only tau x{r}: dchi2 (8 beats) = {2*f.cost*8:.3f}')


if __name__ == "__main__":
    grid = [(R, C) for R in (0.7, 1.0, 1.4) for C in (0.9, 1.5, 2.2)]
    print("=" * 78)
    print("STRUCTURAL: smallest and second-smallest singular value (relative),")
    print("and angle between the numerical null vector and the predicted scaling")
    print("=" * 78)
    for cls in ("L0", "L1", "L2"):
        S = structural(cls, grid)
        print(f"{cls}: n = {len(S):2d} parameter sets | s_min/s_max max {S[:,0].max():.1e} | "
              f"s_(p-1)/s_max min {S[:,1].min():.1e} | angle max {S[:,2].max():.3f} deg")

    # explicit invariance check for L2
    a = 1.7
    th = theta("L2", 1.0, 1.5); th2 = th.copy()
    th2[0] += np.log(a); th2[1] -= np.log(a); th2[3] += np.log(a); th2[4] += np.log(a)
    y1, y2 = obs(th, "L2"), obs(th2, "L2")
    print(f"\nL2 explicit invariance, alpha = {a}: max|y2 - y1| / ptp = "
          f"{np.max(np.abs(y2 - y1)) / np.ptp(y1):.1e}")
    th = theta("L1", 1.0, 1.5, 1.3); th2 = th.copy()
    th2[0] += np.log(a); th2[1] -= np.log(a); th2[2] -= 1.3 * np.log(a)
    y1, y2 = obs(th, "L1"), obs(th2, "L1")
    print(f"L1 explicit invariance, alpha = {a}: max|y2 - y1| / ptp = "
          f"{np.max(np.abs(y2 - y1)) / np.ptp(y1):.1e}")

    print("\n" + "=" * 78)
    print("PRACTICAL: profile in tau, known inflow, full waveform, noise 1% of pulse")
    print("amplitude, 8 beats at 500 Hz. dchi2 > 3.84 excludes the value at 95%.")
    print("=" * 78)
    for cls in ("L0", "L1", "L2"):
        for R, C in [(1.0, 1.5), (1.4, 0.9)]:
            pr = profile_tau(cls, R, C)
            txt = "  ".join(f"{r:.2f}:{d:9.2f}" for r, d in pr)
            print(f"{cls} tau={R*C:.2f}  {txt}")

    print("\n" + "=" * 78)
    print("DIASTOLIC DECAY ONLY (0.36 s, Q = 0): singular values and L2 profile in tau")
    print("=" * 78)
    diastolic_report()
