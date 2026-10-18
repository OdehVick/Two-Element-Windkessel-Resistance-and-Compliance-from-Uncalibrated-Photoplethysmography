[README.md](https://github.com/user-attachments/files/33617597/README.md)
# Two-element Windkessel resistance and compliance from uncalibrated photoplethysmography

Code for the article

> O. V. Adeyi, X. Zhang, A. Aligayev, M. A. Ediete, O. Olamide, A. Clement.
> *Structural Non-Identifiability of Two-Element Windkessel Resistance and Compliance from Uncalibrated Photoplethysmography.*
> Manuscript under review, *Computers in Biology and Medicine* (CIBM-D-26-06085).

## What this study shows

An uncalibrated photoplethysmogram (PPG) cannot identify the resistance *R* and compliance *C* of a two-element Windkessel separately. An unknown optical gain absorbs the rescaling (*R*, *C*) → (α*R*, *C*/α), so only the time constant τ = *RC* can be identified, and from the diastolic decay only if the optical transduction is affine. On 3,459 paired intraoperative arterial-pressure and PPG recordings from [VitalDB](https://vitaldb.net), the estimated τ depended strongly on the assumed diastolic asymptote, and after allowing for measurement noise most recordings failed a mono-exponential adequacy test. The repository contains the simulator, the identifiability analyses, the clinical pipeline and every sensitivity analysis reported in the article and its supplement.

## Repository layout

```
code/        all analysis code (flat, so the scripts can import each other)
archive/     superseded or exploratory scripts, not needed for the article
requirements.txt
LICENSE
CITATION.cff
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/OdehVick/Two-Element-Windkessel-Resistance-and-Compliance-from-Uncalibrated-Photoplethysmography.git
cd Two-Element-Windkessel-Resistance-and-Compliance-from-Uncalibrated-Photoplethysmography
pip install -r requirements.txt
cd code          # run every script from inside this folder
```

The synthetic analyses run on any machine. The clinical analyses download waveforms from VitalDB through the `vitaldb` package and therefore need internet access; the full cohort takes several hours, so we ran them on Google Colab with results written to Google Drive. Every script that writes a file accepts `--out` or `--outdir`.

## Data

All clinical data come from the open-access [VitalDB](https://vitaldb.net) database (Lee et al., *Scientific Data* 2022). No data are redistributed here. Cases are selected by track name inside the code (`SNUADC/ART`, `SNUADC/PLETH`, and a cardiac-output track for the reference cohort), so the cohorts are rebuilt exactly from the public database. Random seeds are fixed in each script.

## Core modules

These are imported by the analysis scripts.

| Module | Role | Article |
|---|---|---|
| `wk_sim.py` | Two-element Windkessel forward simulator; transduction classes L0 (affine), L1 (power law), L2 (Langewouters volumetric); additive noise | §2.4–2.5 |
| `vitaldb_pipeline.py` | Cohort selection, waveform loading, signal-quality gate, beat (foot) detection | §2.2–2.3 |
| `tau_fix.py` | Deterministic diastolic-onset rule | §2.3 |
| `tau_multibeat.py` | Joint eight-beat estimator of τ with a shared asymptote and asymptote constraints | §2.8–2.9 |
| `identifiable_estimator.py` | Split-window adequacy diagnostic and the estimator that reports only identifiable quantities (originally prespecified rule) | §2.10–2.11 |
| `snr_est.py` | Signal-to-noise ratio from the 40–200 Hz spectral noise floor | §2.10 |

## Reproducing the article

### Theory and synthetic data (no internet needed)

| Script | Produces | Article |
|---|---|---|
| `verify_theory.py` | Exactness of the scaling equivalence under L0, L1 and L2 | §3.1, Table S2 |
| `identifiability_hierarchy.py` | Full-waveform structural and practical identifiability with known inflow: sensitivity rank, null direction, profile χ² for τ | §2.6, Table 6 |
| `estimator_experiment.py` | Virtual-cohort experiment: a learned estimator of *R* with a subject-level holdout, and its response along the non-identifiable direction | §2.12, §3.6 |
| `threshold_validation.py` | Justification of the spread threshold (1.5) and the split-window tolerance on synthetic waveforms | §2.9–2.10, S1 |
| `noise_validation.py` | Behaviour of the adequacy diagnostic under white noise | S1 |
| `det_cal.py` | Detection and false-decline rates of the noise-calibrated rule (Windkessel, reflection, L2) | S1 |
| `fp200.py` | False-decline rate of the noise-calibrated rule at 30, 33 and 36 dB (≈ 11.5% at 33 dB) | §3.11 |

```bash
python verify_theory.py
python identifiability_hierarchy.py
python estimator_experiment.py
python threshold_validation.py
python noise_validation.py --skip-clinical
python det_cal.py
python fp200.py
```

### Clinical analyses (VitalDB)

Run in this order; later scripts read the CSV files written by earlier ones.

| Step | Command | Output | Article |
|---|---|---|---|
| 1 | `python run_spread.py --outdir results` | `results_spread.csv` | §3.8 (constraint dependence, Figure 4) |
| 2 | `python run_c4_validation.py --outdir results` | `results_c4.csv` | §3.11 (adequacy cohort) |
| 3 | `python run_expb.py --outdir results` | `results_expb.csv` | §3.10 (reference cohort) |
| 4 | `python fix_expb.py --csv results/results_expb.csv` | corrected §3.10 statistics | §3.10 |
| 5 | `python analyse_spread.py --csv results/results_spread.csv` | mechanism tests, channel agreement | §3.8.1, §3.9 |
| 6 | `python figures_clinical.py --csv results/results_spread.csv --outdir figs` | clinical figures | Figures 4–6, 8 |
| 7 | `python device_and_demographics.py --spread results/results_spread.csv --expb results/results_expb.csv --c4 results/results_c4.csv` | cohort table, device breakdown, Figure 2 counts | §2.2, §3.10 |
| 8 | `python extra_bounds.py --spread results/results_spread.csv --out results/results_extra_bounds.csv` | other asymptote bounds; scale-invariant PPG bounds | §3.8 |
| 9 | `python segmentation_sensitivity.py --c4 results/results_c4.csv --out results/results_segmentation.csv` | sensitivity to onset rule, filtering and segment selection | §2.3, §3.11 |
| 10 | `python beats_window.py --c4 results/results_c4.csv --out results/results_beats.csv` | number of beats and window length | §2.8 |
| 11 | `python clinical_snr.py --c4 results/results_c4.csv --out results/results_snr_full.csv` | per-recording SNR | §3.11 |
| 12 | `python noise_calibrated.py --c4 results/results_c4.csv --snr results/results_snr_full.csv` | noise-calibrated adequacy | §3.11 |
| 13 | `python revised_table15.py --c4 results/results_c4.csv --snr results/results_snr_full.csv --spread results/results_spread.csv` | Table 15 | §3.11 |
| 14 | `python band_coverage.py --expb results/results_expb.csv --bounds results/results_extra_bounds.csv` | admissible (*R*, *C*) band and reference coverage | §2.11, §3.10 |
| 15 | `python cvp_sensitivity.py --expb results/results_expb.csv --out results/results_cvp.csv` | effect of central venous pressure on the reference | §3.10 |
| 16 | `python make_waveform_figure.py --csv results/results_c4.csv --out figs/fig_waveforms.png` | example waveforms | Figure 7 |
| 17 | `python threshold_sweep.py --csv results/results_c4.csv` | pass rate under the original zero-acceleration rule | Table S6 |
| 18 | `python imputation_sensitivity.py --out results/results_imputation.csv` | effect of missing-sample imputation | Table S7 |
| 19 | `python confidence_intervals.py --spread results/results_spread.csv --expb results/results_expb.csv --c4 results/results_c4.csv` | bootstrap and Wilson intervals | throughout |

Most clinical scripts accept `--n` (or `--pilot`) to run on a small number of cases first.

## Adequacy rule used in the article

`identifiable_estimator.py` implements the originally prespecified rule: a recording is declined if any beat shows accelerating decay or the late/early τ ratio falls outside 1/1.3–1.3. Because measurement noise alone produces accelerating beats, the article reports the **noise-calibrated** rule implemented in `noise_calibrated.py` and `revised_table15.py`: a recording is declined if its number of accelerating beats exceeds the noise expectation at its own SNR (one-sided binomial test, *p* < 0.05) or its late/early τ ratio falls outside 1/2.0–2.0. Results under the original rule are kept for comparison (Table S6).

## Correction note

An earlier version of `noise_validation.py` applied the diagnostic to the noise-free simulated pressure rather than to the noisy observation. This is corrected here, and the correction is disclosed in the article and in supplementary section S1.

## Citation

Please cite the article (see `CITATION.cff`). The citation will be updated with the journal reference on publication.

## License

MIT; see `LICENSE`.

## Contact

Odeh Victor Adeyi, School of Information and Communication Engineering, University of Electronic Science and Technology of China.
