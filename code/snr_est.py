"""snr_est.py — white-noise-equivalent SNR of a recording, from the 40-200 Hz
spectral noise floor (Section 2.10). Imported by det_cal.py."""
import numpy as np
from scipy.signal import welch
def snr_db(x, fs=500.0, band=(40.0, 200.0)):
    """White-noise-equivalent SNR, matching wk_sim.add_noise: AC signal power over
    noise power, with the noise level read from the PSD floor in `band`, where the
    pulse waveform has negligible content, and extrapolated to the full band."""
    x = np.asarray(x, float); x = x - x.mean()
    f, p = welch(x, fs=fs, nperseg=4096)
    m = (f >= band[0]) & (f <= band[1])
    noise_var = np.median(p[m]) * (fs / 2.0)
    sig_var = max(np.var(x) - noise_var, 1e-30)
    return 10 * np.log10(sig_var / noise_var)
