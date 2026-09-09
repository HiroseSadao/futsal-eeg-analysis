# Analysis code for "Pre-movement sensor-level EEG activity recorded with a portable three-channel system is associated with futsal free-kick outcomes"

## Scripts

- `code/00_capture_svm_design.py`: record SVM trial order and permutation seeds.
- `code/01_preprocess_eeg.py`: filter EEG and extract pre-movement segments.
- `code/02_fooof_exclusion.py`: identify trials for exclusion using FOOOF.
- `code/03_svm_analysis.py`: within-day and cross-day SVM analyses.
- `code/04_band_power_analysis.py`: compare alpha- and beta-band power between outcomes.
- `code/05_behavior_summary.py`: summarize kick outcomes.
- `code/06_fooof_spectral_summary.py`: fit participant-mean spectra and summarize FOOOF results.
- `code/07_cwt_analysis.py`: compute Morlet-wavelet power.

## Inputs

Study data and metadata are not included. Scripts read EEG recordings and
participant and outcome tables. The SVM analysis also requires fixed trial
orders and permutation seeds in a JSON file, specified with `--design`.

Run scripts individually. Input and output paths are set through command-line
arguments; see each script's `--help`.

## License

[MIT](LICENSE).
