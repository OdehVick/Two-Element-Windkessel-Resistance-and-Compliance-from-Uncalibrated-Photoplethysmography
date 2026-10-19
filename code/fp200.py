"""fp200.py — false-decline rate of the noise-calibrated rule on 200 synthetic
Windkessel waveforms at 30, 33 and 36 dB (the ~11.5% figure in Section 3.11).
    python fp200.py
"""
from det_cal import rule
import numpy as np
for snr in (30,33,36):
    d,a=rule('wk',snr,n=200,seed=1000+snr)
    print(f'WK {snr} dB n=200: combined rule declined {d:.1f}%, excess-acceleration test {a:.1f}%')
