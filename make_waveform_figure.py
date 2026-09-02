"""
make_waveform_figure.py — example waveforms showing accepted and rejected beats.

The reviewer asked for raw arterial and plethysmographic traces with the
accepted and rejected beats marked, so that a reader can see what the adequacy
diagnostic is responding to rather than taking the rejection rate on trust.

The figure has three rows:

  (a) a case the diagnostic ACCEPTS: diastolic decay is mono-exponential, the
      early and late halves give the same time constant, no beat accelerates
  (b) a case REJECTED for a change in slope: late/early ratio outside tolerance
  (c) a case REJECTED for accelerating decay: the decay rate increases within
      diastole, which no exponential can produce

Each row shows the raw signal with detected feet and the fitted diastolic
window shaded, and beside it the log-derivative on which the diagnostic
operates, with the two half-window regression lines drawn.

Run in Colab:
    !python make_waveform_figure.py \
        --csv /content/drive/MyDrive/ppg_ident/results_c4.csv \
        --out /content/drive/MyDrive/ppg_ident/figs/fig_waveforms.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from vitaldb_pipeline import load_waveforms, segment_quality, detect_feet, FS
from tau_fix import diastole_start
from tau_multibeat import _collect_tails

INK, ACC, REJ = "#1f3864", "#2e7d32", "#c0392b"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.dpi": 300, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "legend.frameon": False})


def halves(tail, fs=FS):
    """Return (t, ln(-dP/dt), split index, early slope, late slope)."""
    w = min(int(0.04 * fs) | 1, (tail.size // 2) * 2 - 1)
    if w < 5:
        return None
    sm = savgol_filter(np.asarray(tail, float), w, 3)
    d = np.gradient(sm, 1.0 / fs)
    t = np.arange(tail.size) / fs
    m = d < 0
    if m.sum() < 30:
        return None
    tt, ln = t[m], np.log(-d[m])
    k = max(1, int(0.05 * tt.size))
    tt, ln = tt[k:-k], ln[k:-k]
    if tt.size < 24:
        return None
    h = tt.size // 2
    out = []
    for a, b in ((tt[:h], ln[:h]), (tt[h:], ln[h:])):
        A = np.vstack([a, np.ones_like(a)]).T
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        out.append(c)
    return tt, ln, h, out[0], out[1]


def classify(tails, fs=FS, tol=1.30):
    """'accept' | 'slope' | 'accel' | None."""
    te, tl, n_acc, n = [], [], 0, 0
    for x in tails:
        r = halves(x, fs)
        if r is None:
            continue
        n += 1
        _, _, _, ce, cl = r
        if ce[0] >= 0 or cl[0] >= 0:
            n_acc += 1
            continue
        te.append(-1/ce[0]); tl.append(-1/cl[0])
    if n < 3:
        return None, np.nan, np.nan
    if n_acc > 0:
        return "accel", n_acc / n, np.nan
    ratio = float(np.median(tl) / np.median(te))
    ok = (1/tol) <= ratio <= tol
    return ("accept" if ok else "slope"), 0.0, ratio


def load_case(cid, fs=FS, win_s=60.0):
    v = load_waveforms(cid, fs=fs)
    if v is None or v.shape[0] < fs * 720:
        return None
    mid = v.shape[0] // 2
    half = int(fs * win_s / 2)
    abp, ppg = v[mid-half:mid+half, 0], v[mid-half:mid+half, 1]
    ok, _ = segment_quality(abp, ppg, fs)
    if not ok:
        return None
    return (np.nan_to_num(abp, nan=float(np.nanmedian(abp))),
            np.nan_to_num(ppg, nan=float(np.nanmedian(ppg))))


def find_examples(caseids, need=("accept", "slope", "accel"), scan=300):
    """Scan for one arterial example of each verdict."""
    found = {}
    for k, cid in enumerate(caseids):
        if len(found) == len(need) or k >= scan:
            break
        try:
            got = load_case(int(cid))
        except Exception:
            continue
        if got is None:
            continue
        abp, _ = got
        tails = _collect_tails(abp, detect_feet(abp, FS), diastole_start, FS, n_beats=8)
        if len(tails) < 4:
            continue
        verdict, frac, ratio = classify(tails)
        if verdict in need and verdict not in found:
            found[verdict] = (int(cid), abp, tails, frac, ratio)
            print(f"   {verdict:7s} -> case {cid}  "
                  f"accel {100*frac:.0f}%  ratio {ratio:.2f}")
    return found


def draw(found, out):
    rows = [k for k in ("accept", "slope", "accel") if k in found]
    fig, ax = plt.subplots(len(rows), 2, figsize=(7.2, 2.05*len(rows)),
                           gridspec_kw={"width_ratios": [1.5, 1.0]})
    if len(rows) == 1:
        ax = np.array([ax])

    titles = {
        "accept": "accepted: mono-exponential decay",
        "slope":  "rejected: time constant changes within diastole",
        "accel":  "rejected: decay rate accelerates",
    }
    cols = {"accept": ACC, "slope": REJ, "accel": REJ}

    for r, key in enumerate(rows):
        cid, sig, tails, frac, ratio = found[key]
        col = cols[key]

        # ---- left: raw trace, three beats, diastolic windows shaded
        feet = detect_feet(sig, FS)
        a, b = feet[2], feet[5]
        seg = sig[a:b]
        t = np.arange(seg.size) / FS
        ax[r, 0].plot(t, seg, color=INK, lw=0.9)
        for f0, f1 in zip(feet[2:5], feet[3:6]):
            beat = sig[f0:f1]
            i0 = diastole_start(beat, FS)
            if i0 is None:
                continue
            s0 = (f0 - a + i0) / FS
            s1 = (f1 - a) / FS
            ax[r, 0].axvspan(s0, s1, color=col, alpha=0.13, lw=0)
        for f in feet[2:6]:
            ax[r, 0].plot((f-a)/FS, sig[f], "v", color=col, ms=3.5)
        ax[r, 0].set_ylabel("ABP (mmHg)")
        ax[r, 0].set_title(f"({'abc'[r]}) case {cid} \u2014 {titles[key]}", loc="left",
                           color=col)
        if r == len(rows)-1:
            ax[r, 0].set_xlabel("time (s)")

        # ---- right: log-derivative with the two half-window fits
        tl = tails[1] if len(tails) > 1 else tails[0]
        h = halves(tl, FS)
        if h is not None:
            tt, ln, k, ce, cl = h
            ax[r, 1].plot(tt, ln, ".", color="#9aa0a6", ms=2.2)
            ax[r, 1].plot(tt[:k], ce[0]*tt[:k]+ce[1], "-", color=INK, lw=1.4,
                          label=f"early  slope {ce[0]:+.2f}")
            ax[r, 1].plot(tt[k:], cl[0]*tt[k:]+cl[1], "-", color=col, lw=1.4,
                          label=f"late   slope {cl[0]:+.2f}")
            ax[r, 1].legend(fontsize=6.2, loc="best")
        ax[r, 1].set_ylabel(r"$\ln(-dP/dt)$")
        if r == len(rows)-1:
            ax[r, 1].set_xlabel("time into diastole (s)")
        note = (f"accelerating beats {100*frac:.0f}%" if key == "accel"
                else (r"$\tau_{late}/\tau_{early}$ = " + f"{ratio:.2f}"))
        ax[r, 1].set_title(note, loc="right", fontsize=7, color=col,
                           fontweight="normal")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="fig_waveforms.png")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    ok = df[df.loaded == True]
    # search accepted cases first, they are rare
    acc_ids = ok[ok.abp_status.isin(["ok", "weak"])].caseid.tolist()
    rest = ok[~ok.abp_status.isin(["ok", "weak"])].caseid.tolist()
    print(f"searching {len(acc_ids)} candidate accepted cases, "
          f"{len(rest)} others")
    found = find_examples(acc_ids + rest[:300])
    if not found:
        raise SystemExit("no examples found")
    draw(found, a.out)
