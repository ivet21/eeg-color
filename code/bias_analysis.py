import os
import glob
import re
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from mne.decoding import SlidingEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import ttest_rel, ttest_1samp, sem

# ------Configurations------
DATA_DIR = "data"
OUTPUT_DIR = "plots/bias_analysis"
LOG_FILENAME = "loggers/analysis_bias_analysis.log"

TIME_WINDOW = (0.0, 2.5)  # Analysis window - maintenance period
BASELINE = (-0.5, 0.0)  # Baseline correction window - pre-stimuli period


CV_FOLDS = 5  # Increase for robust estimates at cost of speed
RANDOM_STATE = 42  # Change to sample different splits
PLOT_DPI = 300  # DPI for plots, possible from 100(basic draft) to 600(high resolution)
CAPSIZE = 5  # Error bar cap size

# Time windows
TIME_WINDOWS = {
    "encoding": (0.0, 0.5),  # Initial encoding period
    "maintenance": (0.5, 2.0),  # Memory maintenance
    "before_retrieval": (2.0, 2.5),  # Just before retrieval/probe period
}
# Condition codes
CONDITION_CODES = {
    "dark": [1, 4],  # Darker distractor
    "light": [2, 3],  # Lighter distractor
}

# -----Logger Configurations-----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode="w"),  # Overwrite log each run
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)


# ---Helper functions-----
def create_directory(path):
    """
    Ensure that a directory exists, if it is absent, create a new one
    :param path: Path to the directory
    """
    os.makedirs(path, exist_ok=True)
    logger.info(f"Ensured directory: {path}")


def load_erp_epochs(subject_id, task_name):
    """
    Load ERP data for a subject and task into MNE Epochs for easier work with mne library
    Also add all necessary metadata for this goal
    :param subject_id: ID of subject (for example subject number 2)
    :param task_name: name of the task ('search' or 'memory')
    :return: Epochs object with the eeg data and corresponding metadata
    """
    # Finding the Matlab file in the folder file corresponding to the subject and taking the data for the right task
    filename = os.path.join(DATA_DIR, f"erp_fixed{subject_id}.mat")
    matlab_data = loadmat(filename, squeeze_me=True, struct_as_record=False)
    task_struct = matlab_data[f"ERP_{task_name}{subject_id}"]

    # Setting the information for the frequencies at a time point
    data_array = np.stack(task_struct.trial, axis=0)  # (n_epochs, n_channels, n_times)
    time_points = np.array(task_struct.time)
    sampling_freq = 1.0 / np.diff(time_points).mean()

    # Setting the information for the channels with their labels, position and frequencies
    channel_names = [str(lbl) for lbl in task_struct.label]
    info = mne.create_info(channel_names, sampling_freq, "eeg")
    info.set_montage("biosemi128")
    epochs = mne.EpochsArray(data_array, info, tmin=time_points[0])

    # Setting the information
    trial_info = task_struct.trialinfo
    color = [ti.col for ti in trial_info]
    memory_error = [ti.mem_err for ti in trial_info]
    # Set the data frame
    epochs.metadata = pd.DataFrame(
        {
            "color": color,
            "mem_err": memory_error,
        }
    )
    return epochs


def apply_preprocessing(epochs):
    """
    Apply baseline correction
    :param epochs: Raw Epochs that are load into this file
    :return: Preprocessed Epochs with the right time and baseline correlations
    """
    return epochs.copy().apply_baseline(BASELINE)


# ------Decoder Factory--------
def build_decoder():
    """
    Build a sliding-window ridge regression decoder
    :return: SlidingEstimator
    """
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    return SlidingEstimator(pipe, n_jobs=1)


# ------Decoder-------
def decode_bias(subjects, windows, condition_codes=None):
    """
    Decode memory error bias for given windows and conditions, if provided
    :param subjects: List of subject IDs (ID example subject number 2)
    :param windows: Time windows of interest (encoding, maintenance and retrieval period)
    :param condition_codes: If provided, compute per condition; else overall
    :return: biases
    """
    # Create biases, if condition is provided, per condition
    if condition_codes:
        bias = {c: {w: [] for w in windows} for c in condition_codes}
    else:
        bias = {w: [] for w in windows}
    # Build decoder and fit the data
    decoder = build_decoder()
    for subj in subjects:
        logger.info(f"Processing subject {subj}")
        epochs = mne.concatenate_epochs(
            [load_erp_epochs(subj, t) for t in ["search", "memory"]]
        )
        epochs = apply_preprocessing(epochs)
        X = epochs.get_data()
        y = epochs.metadata["mem_err"].values
        conditions = epochs.metadata["color"].values
        predictions = decoder.fit(X, y).predict(X)
        times = epochs.times

        # Calculate the errors and the mean
        if condition_codes:
            for cond_label, codes in condition_codes.items():
                # If there is a condition add a mask
                mask = np.isin(conditions, codes)
                for window_name, (t_start, t_end) in windows.items():
                    i_start, i_end = np.searchsorted(times, (t_start, t_end))
                    window_means = predictions[mask, i_start:i_end].mean(axis=1)
                    errors = window_means - y[mask]
                    logger.info(
                        f"Subject {subj},cond={cond_label}, window={window_name} with mean={window_means.mean()} and bias={errors.mean():.3f}"
                    )
                    bias[cond_label][window_name].append(errors.mean())
        else:
            for window_name, (t_start, t_end) in windows.items():
                i_start, i_end = np.searchsorted(times, (t_start, t_end))
                window_means = predictions[:, i_start:i_end].mean(axis=1)
                errors = window_means - y
                logger.info(
                    f"Subject {subj}, window={window_name} with mean={window_means.mean()}"
                )
                bias[window_name].append(errors.mean())

    return bias


# --------Plots---------
def analyze_and_plot(bias, windows, conditions):
    """
    Perform statistical tests and generate bar plots for biases
    :param bias: decoded biases
    :param windows: Time windows mapping
    :param conditions: True if bias keyed by condition
    :return: None
    """
    # Generate in one plot the data for condition dark and light by window
    if conditions:
        for w in windows:
            dark = np.array(bias["dark"][w])
            light = np.array(bias["light"][w])
            # t - how spread-out the values are, standard error
            # p - probability that true mean difference is zero
            t, p = ttest_rel(
                light, dark
            )  
            logger.info(
                f"{w}: dark={dark.mean():.3f}, light={light.mean():.3f}, t={t:.2f}, p={p:.3f}"
            )
            plt.figure()
            plt.bar(
                ["dark", "light"],
                [dark.mean(), light.mean()],
                yerr=[sem(dark, ddof=1), sem(light, ddof=1)],
                capsize=CAPSIZE,
            )
            plt.ylabel("Bias (°)")
            plt.title(f"{w}:  t={t:.2f}, p={p:.3f}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"bias_{w}.svg"), dpi=PLOT_DPI)
            plt.close()
    else:
        # If condition is not provided show all the 3 time periods and ist data next to each other
        for w in windows:
            # t - how spread-out the values are, standard error
            # p - probability that the average is at zero
            values = np.array(bias[w])
            t, p = ttest_1samp(
                values, 0
            ) 
            logger.info(f"{w}: t={t:.2f}, p={p:.3f}")

        # Setting the additional information
        labels = list(windows)
        means = [np.mean(bias[l]) for l in labels]
        sems = [
            np.std(bias[w], ddof=1) / np.sqrt(len(bias[w])) for w in labels
        ]  # Standard devotion
        # Format the plot
        plt.figure()
        plt.bar(labels, means, yerr=sems, capsize=CAPSIZE)
        plt.axhline(0, linestyle="--")
        plt.ylabel("Bias (°)")
        plt.title("Overall bias")

        # Save as svg for better quality
        plt.savefig(os.path.join(OUTPUT_DIR, "overall_bias.svg"), dpi=PLOT_DPI)
        plt.close()


# -----Main Script-------------
if __name__ == "__main__":
    """
    Main execution: load data, decode, test and plot
    """
    create_directory(OUTPUT_DIR)

    files = glob.glob(os.path.join(DATA_DIR, "erp_fixed*.mat"))
    subjects = sorted(
        {
            re.match(r"erp_fixed(\d+)\.mat$", os.path.basename(f)).group(1)
            for f in files
            if re.match(r"erp_fixed(\d+)\.mat$", os.path.basename(f))
        }
    )

    # Time-resolved hue bias by condition
    bias_with_condition = decode_bias(subjects, TIME_WINDOWS, CONDITION_CODES)
    analyze_and_plot(bias_with_condition, TIME_WINDOWS, True)

    # Overall hue bias
    bias_all = decode_bias(subjects, TIME_WINDOWS)
    analyze_and_plot(bias_all, TIME_WINDOWS, False)

    logger.info("Done with this script")
