# Setup and Guide

# Introduction

This repository contains analysis scripts for an EEG-based experiment, divided into two tasks: search and memory. It provides three separate Python modules that load ERP and time–frequency data, apply preprocessing, perform decoding and regression analyses, conduct statistical testing, and generate representative plots.

# Requirements
**Python Version**: 3.9 or higher
**Libraries**: 
Managed via conda environment (environment.yml)
  - numpy 
  - scipy 
  - pandas 
  - matplotlib 
  - scikit-learn 
  - mne 
  - pip

## Installation

Use the provided Conda environment configuration to set up all dependencies:
```
    conda env create -f environment.yml
    conda activate eeg-analysis
```

That’s it—no need for manual package installs since environment.yml handles everything.

# Running the Analyses

Ensure your current directory is the project root, the data/ folder contains the required .mat files, and the EEG-analysis environment is active.

## 0. Preprocess MATLAB `.mat` Files

Before running any Python code, convert the raw ERP/TFR files into “fixed” versions that:
- Split each subject’s data into memory vs. search structs  
- Turn `trialinfo` tables into plain structs  
- Save one ERP and one TF `.mat` per subject

**Example for the matlab:**
```
% load ERP and TF for subject 2
erp2 = load(fullfile(basePath, 'OT_S2_singletrial_ERP.mat'));
tf2  = load(fullfile(basePath, 'OT_S2_singletrial_TF.mat'));

% process ERP
ERP_memory2 = erp2.ERP_memory;
ERP_search2 = erp2.ERP_search;
ERP_memory2.trialinfo = table2struct(ERP_memory2.trialinfo);
ERP_search2.trialinfo = table2struct(ERP_search2.trialinfo);
save(fullfile(baseFolder, 'erp_fixed2.mat'), 'ERP_memory2', 'ERP_search2', '-v7');

% process TF
data_TF_memory2 = tf2.data_TF_memory;
data_TF_search2 = tf2.data_TF_search;
data_TF_memory2.trialinfo = table2struct(data_TF_memory2.trialinfo);
data_TF_search2.trialinfo = table2struct(data_TF_search2.trialinfo);
save(fullfile(baseFolder, 'tf_fixed2.mat'), 'data_TF_memory2', 'data_TF_search2', '-v7');
```

## 1. Color Decoder
```
    python code/color_decoder.py
```
 - Description: Applies CSP and SPoC spatial filters to ERP data for brightness and difficulty decoding across ROIs.
 - Outputs:
    - Logs: loggers/analysis_color_decoder.log
    - Plots: plots/color_decoder/summary_brightness.svg, plots/color_decoder/summary_difficulty.svg

## 2. Bias Analysis
```
    python code/bias_analysis.py
```
 - Description: Time-resolved ridge regression to decode memory error bias  with or without a condition, during encoding, maintenance, and retrieval.
 - Outputs:
	 - Logs: loggers/analysis_bias_analysis.log
	 - Plots:
	 - plots/`bias_analysis/bias_encoding.svg`
	 - plots/`bias_analysis/bias_maintenance.svg`
	 - plots/`bias_analysis/bias_retrieval.svg`
	 - plots/`bias_analysis/overall_bias.svg`

## 3. Memory Error Analysis
```
    python code/memory_error_decoder.py
```
- Description: Runs multiple regression/classification pipelines to determine memory error:
1. Sliding ERP analyses
2. Sliding TFR analyses
3. Static CSP regression
4. Static SPoC regression
- Outputs:
     - Logs: loggers/analysis_memory_error.log
     - Plots: SVGs under plots/memory_error/ organized by ROI and method.

## Common Parameters
Modify these at the top of each script:
 - Directories:
     - `DATA_DIR` - Data folder (default: data) 
     - `OUTPUT_DIR` - Output folder (plots/color_decoder)
	 - `LOG_FILENAME` - Logger file name
 - Time Settings:
	 - `TIME_WINDOW` - Time windows of interest (encoding, maintenance and retrieval period)
	 - `BASELINE` - Baseline correction window - pre-stimuli period
 - Cross-Validation:
	 - `CV_FOLDS` - increase for robust estimates at cost of speed (default: 5)
	 - `RANDOM_STATE` - change to sample different splits(default: 42)
 - Plotting: 
	 - `PLOT_DPI` - DPI for plots, possible from 100(basic draft) to 600(high resolution)(default: 300)
	 - `CAPSIZE` - Error bar cap size (error bar cap size)
 - Analysis-Specific:
	 - CSP/SPoC components 
        - `CSP_COMPONENTS` - increase to capture more variance, decrease to avoid overfitting (default: 10)
        - `SPOC_COMPONENTS` - fewer components = simpler model, more robust to noise (default: 4)
	 - Frequency bands (`BANDS`) for TFR analysis
	 - Condition mappings (`CONDITION_GROUPS`, `CONDITION_CODES`)

## Code Explanation

### Data Loading
 - ERP Data: `load_erp_epochs()` (or `DataHandler.load_erp()`) loads .mat files into MNE Epochs, attaching metadata.
 - TFR Data: `DataHandler.load_tfr()` loads time–frequency .mat files into EpochsTFRArray with metadata.

### Preprocessing
 - Baseline correction using defined `BASELINE` window.
 - Cropping epochs to `TIME_WINDOW`.

### Feature Extraction & Decoding
 - Spatial Filters: CSP and SPoC via SpatialFilterClassifier.
 - Regression & Classification: Ridge, SVR, RandomForest pipelines (`build_pipeline()`).
 - Sliding Analysis: Time-resolved decoding with `mne.decoding.SlidingEstimator`.

### Statistical Testing
 - t-tests: One-sample (`ttest_1samp`) and paired (`ttest_rel`).
 - Cluster-Based Permutation: `spatio_temporal_cluster_1samp_test` for time-series significance.

### Plotting
 - Bar Plots: Mean and SEM saved as SVG.
 - Time-Series: Line plots with shaded SEM and significant timepoints or CV fold stats.
