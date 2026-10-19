"""
figures_clinical.py — item 4: publication figures from results_spread.csv.

Run in Colab:
    !python figures_clinical.py --csv /content/drive/MyDrive/ppg_ident/results_spread.csv \
                                --outdir /content/drive/MyDrive/ppg_ident/figs

Figure 2  Prior dependence of tau.
          (a) tau vs prior, per-case trajectories + medians, both channels
          (b) fraction of fits attaining the bound, by prior
          The monotone decline of (b) is the signature of a likelihood with no
          interior optimum in the physiological region.

Figure 3  Fit quality is not evidence of parameter validity.
          (a) r2 vs estimated tau, unconstrained fit, arterial line, with the
              plausible region shaded; the 76.0% of cases with r2 > 0.95 and
              implausible parameters occupy the region outside it
          (b) same for the plethysmogram

Figure 4  The mechanism is model mismatch, not short-window degeneracy.
          (a) spread vs diastolic window length, with tertile medians. The
              association is POSITIVE: longer windows give WORSE prior
              dependence, contrary to a degeneracy explanation
          (b) spread distribution, both channels, against the synthetic
              reference value of 1.00

Figure 5  Transduction exponent across the cohort.
          Distribution of tau_PPG / tau_ABP, which estimates 1/gamma. L0
          predicts exactly 1 for every patient; only ~9% of the cohort falls
          within 10% of that value.
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIORS = [("tight", "(0,10)"), ("physio", "(0,25)"), ("wide", "(0,40)"),
          ("permissive", "(-20,60)"), ("free", "unbounded")]
TAU_LO, TAU_HI = 0.15, 12.0
SPREAD_MAX = TAU_HI / TAU_LO
ABP_C, PPG_C = "#1f3864", "#c0392b"

plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 200, "savefig.dpi": 300,
    "legend.frameon": False, "axes.titlesize": 9, "axes.titleweight": "bold",
})


def num(s):
    return pd.to_numeric(s, errors="coerce")


def railed(ok, pre):
    r = np.zeros(len(ok), dtype=bool)
    for p, _ in PRIORS:
        t = num(ok[f"{pre}_tau_{p}"]).values
        r |= (np.abs(t - TAU_LO) < 1e-3) | (np.abs(t - TAU_HI) < 1e-3)
    return r


# --------------------------------------------------------------------------
def fig2(ok, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.arange(len(PRIORS))

    for pre, col, lab in [("abp", ABP_C, "arterial line"),
                          ("ppg", PPG_C, "plethysmogram")]:
        M = np.column_stack([num(ok[f"{pre}_tau_{p}"]) for p, _ in PRIORS])
        keep = np.isfinite(M).all(axis=1)
        M = M[keep]
        sel = np.random.default_rng(0).choice(len(M), min(250, len(M)),
                                              replace=False)
        for row in M[sel]:
            ax[0].plot(x, row, color=col, alpha=0.045, lw=0.6)
        ax[0].plot(x, np.median(M, axis=0), "o-", color=col, lw=2.0,
                   ms=4.5, label=lab, zorder=5)

    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{p}\n{r}" for p, r in PRIORS], fontsize=6.5)
    ax[0].set_ylabel(r"estimated $\tau$ (s)")
    ax[0].set_xlabel(r"prior on $P_\infty$ (mmHg)")
    ax[0].set_title("(a) Estimated time constant by prior", loc="left")
    ax[0].axhspan(0.5, 4.0, color="#2e7d32", alpha=0.07, zorder=0)
    ax[0].text(0.05, 0.62, "physiological\nrange", fontsize=6,
               color="#2e7d32", transform=ax[0].transAxes)
    ax[0].legend(fontsize=7, loc="upper right")

    fr = [100 * num(ok[f"abp_hit_{p}"].astype(float)).mean() for p, _ in PRIORS]
    ax[1].bar(x, fr, color=ABP_C, width=0.62)
    for xi, v in zip(x, fr):
        ax[1].text(xi, v + 1.8, f"{v:.1f}", ha="center", fontsize=6.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([p for p, _ in PRIORS], fontsize=6.5, rotation=20)
    ax[1].set_ylabel("fits attaining the bound (%)")
    ax[1].set_ylim(0, 108)
    ax[1].set_title("(b) Constraint, not data, arrests the fit", loc="left")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_prior_dependence.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig3(ok, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for a, (pre, col, lab) in zip(ax, [("abp", ABP_C, "arterial line"),
                                       ("ppg", PPG_C, "plethysmogram")]):
        t = num(ok[f"{pre}_tau_free"])
        r2 = num(ok[f"{pre}_r2_free"])
        pi = num(ok[f"{pre}_pinf_free"])
        m = np.isfinite(t) & np.isfinite(r2)
        t, r2, pi = t[m], r2[m], pi[m]

        # Plausibility must match run_spread.py exactly: tau AND P_inf.
        # Judging on tau alone (the plotted axis) understates the failure.
        pl_tau = (t > 0.5) & (t < 4.0)
        pl_full = pl_tau & (pi > -10) & (pi < 40)
        bad_tau = (r2 > 0.95) & ~pl_tau
        bad_full = (r2 > 0.95) & ~pl_full
        hidden = bad_full & ~bad_tau          # implausible only via P_inf

        a.scatter(t[~bad_full], r2[~bad_full], s=2.5, c="#9e9e9e",
                  alpha=0.35, lw=0)
        a.scatter(t[bad_tau], r2[bad_tau], s=2.5, c=col, alpha=0.45, lw=0)
        a.scatter(t[hidden], r2[hidden], s=3.5, c="none", edgecolors=col,
                  linewidths=0.35, alpha=0.55)
        a.axvspan(0.5, 4.0, color="#2e7d32", alpha=0.09, zorder=0)
        a.axhline(0.95, color="k", ls="--", lw=0.8)
        a.set_xscale("log")
        a.set_xlim(0.1, 14)
        a.set_ylim(0.55, 1.005)
        a.set_xlabel(r"estimated $\tau$ (s), unconstrained fit")
        a.set_ylabel(r"$R^2$")
        a.set_title(f"({'a' if pre=='abp' else 'b'}) {lab}", loc="left")
        a.text(0.97, 0.03, "vertical stripes:\nfits at the "
               r"$\tau$" " bounds", transform=a.transAxes,
               fontsize=6, ha="right", color="#555555")
        a.text(0.03, 0.11,
               f"$R^2>0.95$ and implausible:\n"
               f"{100*bad_full.mean():.1f}% of cases\n"
               f"({100*bad_tau.mean():.1f}% on "
               r"$\tau$" f" alone; +{100*hidden.mean():.1f}% via "
               r"$P_\infty$" ")",
               transform=a.transAxes, fontsize=6.8, color=col)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_fit_vs_plausibility.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig4(ok, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))

    sub = ok[~railed(ok, "abp")].copy()
    d = num(sub["abp_dias_s"]); s = num(sub["abp_spread"])
    m = np.isfinite(d) & np.isfinite(s)
    d, s = d[m], s[m]
    ax[0].scatter(d, s, s=2.5, c=ABP_C, alpha=0.22, lw=0)
    ter = pd.qcut(d, 3, labels=False)
    for k, lab in enumerate(["short", "mid", "long"]):
        ax[0].plot(d[ter == k].median(), s[ter == k].median(), "o",
                   color="#f39c12", ms=8, mec="k", mew=0.7, zorder=6)
        ax[0].annotate(f"{lab}\n{s[ter==k].median():.2f}x",
                       (d[ter == k].median(), s[ter == k].median()),
                       textcoords="offset points", xytext=(9, 4), fontsize=6.5)
    from scipy import stats as st
    rho, p = st.spearmanr(d, s)
    ax[0].axhline(1.0, color="k", ls=":", lw=0.9)
    ax[0].text(0.03, 0.93, f"Spearman $\\rho$ = {rho:+.3f}  (p = {p:.1e})\n"
                           f"longer window $\\rightarrow$ WORSE prior dependence",
               transform=ax[0].transAxes, fontsize=7, va="top")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("diastolic window (s)")
    ax[0].set_ylabel(r"prior spread, $\max\tau/\min\tau$")
    ax[0].set_title("(a) Not short-window degeneracy", loc="left")

    bins = np.logspace(0, np.log10(SPREAD_MAX), 45)
    for pre, col, lab in [("abp", ABP_C, "arterial line"),
                          ("ppg", PPG_C, "plethysmogram")]:
        v = num(ok[f"{pre}_spread"]).dropna()
        ax[1].hist(v, bins=bins, histtype="step", lw=1.5, color=col, label=lab)
    ax[1].axvline(1.0, color="k", ls="--", lw=1.1)
    ax[1].text(1.06, ax[1].get_ylim()[1] * 0.86,
               "synthetic\nWindkessel\n= 1.00", fontsize=6.5)
    ax[1].axvline(1.5, color="#2e7d32", ls=":", lw=1.1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"prior spread, $\max\tau/\min\tau$")
    ax[1].set_ylabel("cases")
    ax[1].set_title("(b) Distribution of prior dependence", loc="left")
    ax[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_mechanism.png"), bbox_inches="tight")
    plt.close(fig)


def fig5(ok, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    keep = (~railed(ok, "abp")) & (~railed(ok, "ppg"))
    sub = ok[keep]

    a = num(sub["abp_tau_free"]); b = num(sub["ppg_tau_free"])
    m = np.isfinite(a) & np.isfinite(b) & (a > 0)
    ratio = (b[m] / a[m]).replace([np.inf, -np.inf], np.nan).dropna()

    bins = np.logspace(np.log10(0.05), np.log10(20), 55)
    ax[0].hist(ratio, bins=bins, color="#7f8c8d", alpha=0.85)
    ax[0].axvline(1.0, color="k", ls="--", lw=1.4)
    ax[0].axvspan(0.9, 1.1, color="#2e7d32", alpha=0.18)
    frac = float(np.mean((ratio > 0.9) & (ratio < 1.1)))
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\tau_{\mathrm{PPG}} / \tau_{\mathrm{ABP}}$  ($=1/\gamma$)")
    ax[0].set_ylabel("cases")
    ax[0].set_title("(a) Affine transduction predicts 1.0", loc="left")
    ax[0].text(0.03, 0.90,
               f"within 10% of unity:\n{100*frac:.1f}% of cases (n = {len(ratio)})",
               transform=ax[0].transAxes, fontsize=7, va="top")

    fr = []
    for p, _ in PRIORS:
        aa = num(sub[f"abp_tau_{p}"]); bb = num(sub[f"ppg_tau_{p}"])
        mm = np.isfinite(aa) & np.isfinite(bb) & (aa > 0)
        r = (bb[mm] / aa[mm]).replace([np.inf, -np.inf], np.nan).dropna()
        fr.append(100 * float(np.mean((r > 0.9) & (r < 1.1))))
    x = np.arange(len(PRIORS))
    ax[1].bar(x, fr, color="#7f8c8d", width=0.62)
    for xi, v in zip(x, fr):
        ax[1].text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=6.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([p for p, _ in PRIORS], fontsize=6.5, rotation=20)
    ax[1].set_ylabel("cases consistent with L0 (%)")
    ax[1].set_ylim(0, max(fr) * 1.35)
    ax[1].set_title("(b) Robust to the choice of prior", loc="left")

    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(os.path.join(outdir, "fig5_transduction.png"),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="./figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.csv)
    ok = df[df.ok == True].reset_index(drop=True)
    print(f"{len(ok)} analysable cases")

    fig2(ok, a.outdir); print("  fig2_prior_dependence.png")
    fig3(ok, a.outdir); print("  fig3_fit_vs_plausibility.png")
    fig4(ok, a.outdir); print("  fig4_mechanism.png")
    fig5(ok, a.outdir); print("  fig5_transduction.png")
    print(f"\nsaved to {a.outdir}")
