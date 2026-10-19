"""det_cal.py — detection and false-decline rates of the noise-calibrated
adequacy rule on synthetic waveforms at the clinical SNR (33 dB): Windkessel
(should pass), reflection r = 0.1-0.3 and volumetric L2 transduction (should fail).
A recording is declined if its count of accelerating beats exceeds the noise
expectation (one-sided binomial test, p < 0.05) or its late/early tau ratio lies
outside 1/2.0-2.0.  Supplementary section S1.
    python det_cal.py
"""
import numpy as np, threshold_validation as tv, segmentation_sensitivity as ss
from scipy.stats import binom
from snr_est import snr_db
REF_SNR=np.array([20,25,30,35,40,45,50.]); REF=np.array([68.5,49.9,32.7,10.9,0.9,0,0])/100
def rule(kind,snr,r=0.0,n=60,seed=0):
    rng=np.random.default_rng(seed); dec=acc=tot=0
    while tot<n:
        sig=tv.make_signal(kind,rng,snr,r); tails,_=ss.collect_tails_p(sig,ss.BASE)
        if len(tails)<4: continue
        _,_,ratio,_,frac=ss.make_split_test(0.04)(tails,500.)
        if not np.isfinite(frac): continue
        nb=len([t for t in tails if t.size>=60]); k=round(frac*nb)
        p0=max(np.interp(snr_db(sig),REF_SNR,REF),1e-4)
        a=binom.sf(k-1,nb,p0)<0.05
        s=(not a) and np.isfinite(ratio) and not (0.5<=ratio<=2.0)
        acc+=a; dec+=a or s; tot+=1
    return 100*dec/n,100*acc/n
if __name__=="__main__":
  for snr in (33,):
      print('WK',snr,'dB: declined %.1f%% (accel %.1f%%)'%rule('wk',snr,seed=1))
      for r in (0.1,0.2,0.3): print('reflection r=%.1f'%r,snr,'dB: declined %.1f%% (accel %.1f%%)'%rule('reflection',snr,r,seed=int(10*r)+2))
      print('L2',snr,'dB: declined %.1f%% (accel %.1f%%)'%rule('L2',snr,seed=9))
