import os
import glob
import re
import logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_1samp, sem
import matplotlib.pyplot as plt
import mne
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.decoding import CSP, SPoC

# ------Configurations------
DATA_DIR = "data"
OUTPUT_DIR = "plots/color_decoder"
LOG_FILENAME = "loggers/analysis_color_decoder.log"

TIME_WINDOW = (0.0, 2.5)  # Analysis window - maintenance period
BASELINE = (-0.5, 0.0)  # Baseline correction window - pre-stimulation period

CSP_COMPONENTS = 10  # Increase to capture more variance, decrease to avoid overfitting
SPOC_COMPONENTS = 4  # Fewer components = simpler model, more robust to noise
CV_FOLDS = 5  # Increase for robust estimates at cost of speed
RANDOM_STATE = 42  # Change to sample different splits

PLOT_DPI = 300  # DPI for plots, possible from 100(basic draft) to 600(high resolution)
CAPSIZE = 5  # Error bar cap size
Y_AXIS_LIMITS = (0.4, 1.0)

# Condition grouping and stimuli mapping
CONDITION_GROUPS = {
    "brightness": {
        "positive": [2, 3],
        "negative": [1, 4],
    },  # Codes: 1=yellow/dark, 2=yellow/light, 3=blue/light, 4=blue/dark
    "difficulty": {
        "positive": [2],
        "negative": [1]
    },  # Codes: 1=hard, 2=easy
}
# Regions of interest defined by channel name prefixes
ROI = {
    "Occipital Region": lambda channels: [
        cha for cha in channels if cha.startswith("A")
    ],  # Channels A*
    "Parietal Region": lambda channels: [
        cha for cha in channels if cha.startswith("B")
    ],  # Channels B*
    "Central Region": lambda channels: [
        cha for cha in channels if cha.startswith("C")
    ],  # Channels C*
    "Frontal Region": lambda channels: [
        cha for cha in channels if cha.startswith("D")
    ],  # Channels D*
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

    # Setting the information for the colors and difficulty of the trails
    trial_info = task_struct.trialinfo
    color = [ti.col for ti in trial_info]
    difficulty = [
        getattr(ti, "diff", np.nan) for ti in trial_info
    ]  # Extract difficulty

    # Set the data frame
    epochs.metadata = pd.DataFrame({"color": color, "difficulty": difficulty})
    return epochs


def apply_preprocessing(epochs):
    """
    Apply baseline correction and crop the data into the right timeframe
    for this project, the maintenance period
    :param epochs: Raw Epochs that are load into this file
    :return: Preprocessed Epochs with the right time and baseline correlations
    """
    return epochs.copy().apply_baseline(BASELINE).crop(*TIME_WINDOW)


# ── CLASSIFIER WRAPPER ───────────────────────────────
class SpatialFilterClassifier(BaseEstimator, TransformerMixin):
    """
    Applies CSP or SPoC spatial filtering
    :param method: 'csp' or 'spoc'
    :param n_components: Number of spatial filter components
    """

    def __init__(self, method, n_components):
        if method not in ("csp", "spoc"):
            raise ValueError(f"Unsupported method: {method!r}")
        self.method = method
        self.n_components = n_components

    def fit(self, X, y):
        """
        Fit spatial filter and pipeline
        :param X: EEG data
        :param y: Labels for the classification
        :return: Self
        """
        self.pipeline = Pipeline(
            [
                ("spatial", self._build_spatial_filter()),
                ("scale", StandardScaler()),
                ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def _build_spatial_filter(self):
        """Instantiate the CSP or SPoC"""
        cls = CSP if self.method.lower() == "csp" else SPoC
        filter = cls(n_components=self.n_components, reg="ledoit_wolf", log=True)
        return filter

    def transform(self, X):
        """
        Transform data using learned spatial filters
        :param X: EEG data to transform
        :return: Array after spatial filtering
        """
        return self.pipeline.named_steps["spatial"].transform(X)

    def predict(self, X):
        """
        Predict class labels for new data
        :param X: EEG data to classify
        :return: Predicted labels y
        """
        return self.pipeline.predict(X)


# -----Decoding--------
def calculate_accuracy(
    epochs, positive_val_codes, negative_val_codes, selector, method
):
    """
    Compute mean accuracy via cross-validation for one ROI with method csp or spoc
    :param epochs: Preprocessed Epochs
    :param positive_val_codes: Condition codes for class 1 that will be part of the mapping in configuration for positive values
    :param negative_val_codes: Condition codes for class 0 that will be part of the mapping in configuration for negative values
    :param selector: Function selecting channels for the specific ROI
    :param method: 'csp' or 'spoc'
    :return: mean classification accuracy
    """
    if method not in ("csp", "spoc"):
        logger.error(f"Method must be 'csp' or 'spoc', got {method!r}")
    # Setting the data
    channels = selector(epochs.ch_names)
    roi_epochs = epochs.copy().pick(channels)
    mask = roi_epochs.metadata["color"].isin(positive_val_codes + negative_val_codes)
    labels = (
        roi_epochs.metadata.loc[mask, "color"]
        .isin(positive_val_codes)
        .astype(int)
        .values
    )
    data = roi_epochs.get_data()[mask]
    if data.size == 0:
        logger.error("No trials found for given codes/ROI.")
    # Creating object from the class and calculating the accuracy
    clf = SpatialFilterClassifier(
        method=method,
        n_components=(SPOC_COMPONENTS if method == "spoc" else CSP_COMPONENTS),
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, data, labels, cv=cv, scoring="accuracy")
    return float(scores.mean())


# --------Plots---------
def plot_metrics(raw_results, group_name):
    """
    Generate and save bar plot of decoding accuracies with standard error of the mean
    :param raw_results: list of mean accuracies for subjects
    :param group_name: type of group ('brightness' or 'difficulty')
    """
    # Setting the information
    methods = list(raw_results.keys())
    data = [np.array(raw_results[m]) for m in methods]
    means = [d.mean() for d in data]
    sems = [sem(d, ddof=1) for d in data]

    # Format of the plot
    fig, ax = plt.subplots()
    bars = ax.bar(methods, means, yerr=sems, capsize=CAPSIZE)
    ax.set_ylim(*Y_AXIS_LIMITS)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{group_name.capitalize()} decoding")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.2f}", ha="center")

    # Save as svg for better quality
    plt.savefig(os.path.join(OUTPUT_DIR, f"summary_{group_name}.svg"), dpi=PLOT_DPI)
    plt.close()


# -----Main Script-------------
if __name__ == "__main__":
    """
    Main execution: load data, preprocess, decode, test, plot and summary statistics
    """
    create_directory(OUTPUT_DIR)
    # Fining all subjects
    files = glob.glob(os.path.join(DATA_DIR, "erp_fixed*.mat"))
    subject_ids = sorted(
        {
            re.match(r"erp_fixed(\d+)\.mat$", os.path.basename(f)).group(1)
            for f in files
            if re.match(r"erp_fixed(\d+)\.mat$", os.path.basename(f))
        }
    )
    n_subjects = len(subject_ids)

    # Initialize storage for group-method-ROI and best-ROI counts
    results = {
        g: {m: {roi: [] for roi in ROI} for m in ["csp", "spoc"]}
        for g in CONDITION_GROUPS
    }
    best_roi_counts = {
        g: {m: Counter() for m in ["csp", "spoc"]} for g in CONDITION_GROUPS
    }

    # Processing the data with csp and spoc
    for subj in subject_ids:
        logger.info(f"Processing subject {subj}")
        epochs = mne.concatenate_epochs(
            [load_erp_epochs(subj, t) for t in ["search", "memory"]]
        )
        epochs = apply_preprocessing(epochs)

        for group_name, codes in CONDITION_GROUPS.items():
            pos_codes, neg_codes = codes["positive"], codes["negative"]
            for method in ["csp", "spoc"]:
                roi_accuracy = []
                for roi_name, selector in ROI.items():
                    acc = calculate_accuracy(
                        epochs, pos_codes, neg_codes, selector, method
                    )
                    results[group_name][method][roi_name].append(acc)
                    roi_accuracy.append((roi_name, acc))
                    logger.info(
                        f"{group_name}/{roi_name} {method.upper()} acc={acc:.3f}"
                    )
                # Log and count best ROI for this subject-group-method
                best_roi, best_acc = max(roi_accuracy, key=lambda x: x[1])
                best_roi_counts[group_name][method][best_roi] += 1
    # Summary statistics and plotting
    for group_name, methods in results.items():
        for method, roi_dict in methods.items():
            all_accuracy = np.concatenate(list(roi_dict.values()))
            t_stat, p_val = ttest_1samp(all_accuracy, 0.5)
            logger.info(
                f"{group_name} {method.upper()} vs chance: t={t_stat:.2f}"
            )
        # Compute subject-level mean per method and plot
        subject_means = {
            m: [np.mean([roi_dict[roi][i] for roi in ROI]) for i in range(n_subjects)]
            for m, roi_dict in methods.items()
        }
        plot_metrics(subject_means, group_name)

    # Log best-ROI counts and overall best results
    for group_name in CONDITION_GROUPS:
        for method in ["csp", "spoc"]:
            counts = best_roi_counts[group_name][method]
            logger.info(
                f"{group_name} {method.upper()} best counts per ROI: {dict(counts)}"
            )
            most_region, most_count = counts.most_common(1)[0]
            mean_acc = np.mean(results[group_name][method][most_region])
            logger.info(
                f"{group_name} {method.upper()} ROI most often best: {most_region} "
                f"({most_count}/{n_subjects}); mean accuracy {mean_acc:.3f}"
            )

    logger.info("Done with this script")
