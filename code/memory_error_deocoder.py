import os
import glob
import re
import logging

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from mne.time_frequency import EpochsTFRArray
from mne.decoding import CSP, SPoC
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.io import loadmat

# ------Configurations------
DATA_DIR = "data"
OUTPUT_DIR = "plots/memory_error"
LOG_FILENAME = "loggers/analysis_memory_error.log"


TIME_WINDOW = (0.0, 2.5)  # Analysis window - maintenance period
BASELINE = (-0.5, 0.0)  # Baseline correction window - pre-stimuli period


CSP_COMPONENTS = 10  # Increase to capture more variance, decrease to avoid overfitting
SPOC_COMPONENTS = 4  # Fewer components = simpler model, more robust to noise
CV_FOLDS = 5  # Increase for robust estimates at cost of speed
RANDOM_STATE = 42  # Change to sample different splits

CV_REG = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
CV_CLF = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

PLOT_DPI = 300  # DPI for plots, possible from 100(basic draft) to 600(high resolution)

# Frequency bands
BANDS = {"theta": (4, 7), "alpha": (8, 12), "beta": (13, 30)}

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


def get_subject_ids(pattern):
    regex = re.compile(pattern)
    files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    ids = {
        regex.search(os.path.basename(f)).group(1)
        for f in files
        if regex.search(os.path.basename(f))
    }
    return sorted(ids)


# ----Data Loaders---------
class DataHandler:
    """
    Load two types of data (erp and tf data ) and then apply baseline and crop the data
    :param baseline: Baseline correction window - pre-stimuli period
    :param window: Analysis window - maintenance period
    """

    def __init__(self, baseline, window):
        self.baseline = baseline
        self.window = window

    @staticmethod
    def load_erp(subject_id, task_name):
        """
        Load ERP data for a given subject and task into MNE Epochs for easier work with mne library
        Also add all necessary metadata for this goal
        :param subject_id: ID of subject (for example subject number 2)
        :param task_name: Name of the task ('search' or 'memory')
        :return: Epochs object with the eeg data and corresponding metadata
        """
        # Finding the Matlab file in the folder file corresponding to the subject and taking the data for the right task
        filename = os.path.join(DATA_DIR, f"erp_fixed{subject_id}.mat")
        matlab_data = loadmat(filename, squeeze_me=True, struct_as_record=False)
        task_struct = matlab_data[f"ERP_{task_name}{subject_id}"]

        # Setting the information for the frequencies at a time point
        data_array = np.stack(
            task_struct.trial, axis=0
        )  # (n_epochs, n_channels, n_times)
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

    @staticmethod
    def load_tfr(subject_id, task_name):
        """
        Load time–frequency (TFR) data for a given subject and task into MNE Epochs for easier work with mne library
        :param subject_id: ID of subject (for example subject number 2)
        :param task_name: Name of the task ('search' or 'memory')
        :return: Epochs object with the eeg data and corresponding metadata
        """
        # Finding the Matlab file in the folder file corresponding to the subject and taking the data for the right task
        filename = os.path.join(DATA_DIR, f"tf_fixed{subject_id}.mat")
        matlab_data = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
        tf_struct = matlab_data[f"data_TF_{task_name}{subject_id}"]

        # Setting the information for the frequencies at a time point
        power_data = np.nan_to_num(np.array(tf_struct.powspctrm))
        frequency = np.array(tf_struct.freq)
        times = np.array(tf_struct.time)
        sampling_freq = 1.0 / np.mean(np.diff(times))

        # Setting the information for the channels with their labels, position and frequencies
        ch_names = [str(ch) for ch in tf_struct.label]
        info = mne.create_info(ch_names=ch_names, sfreq=sampling_freq, ch_types="eeg")
        info.set_montage("biosemi128")
        tfr = EpochsTFRArray(data=power_data, info=info, freqs=frequency, times=times)

        # Setting the information
        trial_info = tf_struct.trialinfo
        mem_errors = [getattr(tr, "mem_err", np.nan) for tr in trial_info]
        # Set the data frame
        tfr.metadata = pd.DataFrame({"mem_err": mem_errors})
        return tfr

    @staticmethod
    def preprocess(epochs):
        """
        Apply baseline correction and crop the data into the right timeframe
        for this project, the maintenance period
        :param epochs: Raw Epochs that are load into this file
        :return: Preprocessed Epochs with the right time and baseline correlations
        """
        return epochs.copy().apply_baseline(BASELINE).crop(*TIME_WINDOW)


# ----Pipeline Factory-------
def build_pipeline(model="ridge"):
    """
    Create pipeline factory for easier access and determine the right method
    :param model: By default ridge but also possible to change depending on the provided information
    :return: Pipeline with the right setup
    """
    if model == "ridge":
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    if model == "svr":
        return Pipeline([("scale", StandardScaler()), ("svr", SVR())])
    if model == "rf":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("rf", RandomForestRegressor(random_state=RANDOM_STATE)),
            ]
        )
    logger.error(f"Unknown model: {model}")


# ----Base class -----------
class BaseAnalysis:
    """
    Base abstract class that generates the main structure of the analysis methods.
    Define the run, load and process, score and test and plot functions
    :param subjects: List if subjects IDs
    :param rois: Regions of interest
    :param handler: Object type DataHandler, that load and process the data
    """

    name = "Base"

    def __init__(self, subjects, rois, handler):
        self.subjects = subjects
        self.rois = rois
        self.handler = handler

    def run(self):
        """
        The process flow of the analyse, describing the main method that will combine all functions stepwise
        """
        if not self.subjects:
            logger.warning(f"No subjects for {self.name}")
            return

        for roi_name, roi_fn in self.rois.items():
            logger.info(f"{self.name} ROI={roi_name}:")
            all_scores = []
            times = None
            for subj in self.subjects:
                data, ts, y, ch_names = self.load_and_preprocess(subj)
                if times is None:
                    times = ts
                feats = self.extract_features(data, roi_fn, ch_names, y)
                score = self.score(feats, y)
                all_scores.append(score)
            arr = np.vstack(all_scores)
            self.test_and_plot(arr, times, roi_name)

    def load_and_preprocess(self, subj):
        """
        Load the data and process it using the DataHandler object
        :param subj: ID of the subject(for example '2')
        """
        raise NotImplementedError

    def extract_features(self, data, roi_fn, ch_names, y):
        """
        Extract the features belonging to Region of interest
        :param data: Erp/tf data form where you pull the data from which to pull ROI channels
        :param roi_fn: Function that take all channels and return iterable of channel names in the ROI
        :param ch_names: All channels that are in the Region of Interest
        :param y: True labels
        """
        raise NotImplementedError

    def score(self, feats, y):
        """
        Compute a score on the extracted features and true labels
        :param feats: Feature matrix (output of extract_features(...))
        :param y: True labels
        """
        raise NotImplementedError

    def test_and_plot(self, arr, times, roi_name):
        """
        Run a cluster-based test on `arr`, plot the mean and SEM over time, mark significant timepoints
        :param arr: array of all scores (output of score(...))
        :param times: time points
        :param roi_name: Name of the specific ROI and if there is a specification of frequency, it also included in the name
        """
        # Run a cluster-based test
        title = f"{self.name} ({roi_name})"
        fname = f"{self.name.lower().replace(' ', '_')}/{roi_name}.svg"
        t_obs, clusters, p_vals, _ = spatio_temporal_cluster_1samp_test(
            arr, n_permutations=200, threshold=0.0, tail=1
        )
        sig = np.zeros(len(times), bool)
        for index, p in zip(clusters, p_vals):
            if p <= 0.05:
                sig[index[0]] = True
        # Setting the information for the plot
        plt.figure()
        mean = arr.mean(0)
        sem = arr.std(0) / np.sqrt(arr.shape[0])
        logger.info(f"{self.name.upper()}/{roi_name} with mean={mean.mean():.3f}")

        create_directory(f"{OUTPUT_DIR}/{self.name.lower().replace(' ', '_')}")
        # Plot set up
        plt.plot(times, mean, label="mean")
        plt.fill_between(times, mean - sem, mean + sem, alpha=0.3)
        plt.scatter(times[sig], np.zeros_like(times[sig]), color="r", s=10)
        plt.title(title)
        plt.axhline(0, linestyle="--", color="k")
        plt.xlabel("Time (s)")
        plt.ylabel("Score")
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=PLOT_DPI)
        plt.close()


# ----- ERP Sliding ------------
class ERPSlidingAnalysis(BaseAnalysis):
    """
    Analysis pipeline that concatenates ERP epochs and evaluate using a ridge regression sliding estimator
    Subclass of Base Analysis
    """

    name = "ERP Sliding R2"

    def load_and_preprocess(self, subj):
        ep = mne.concatenate_epochs(
            [self.handler.load_erp(subj, t) for t in ("search", "memory")]
        )
        ep = self.handler.preprocess(ep)
        return ep.get_data(), ep.times, ep.metadata["mem_err"].values, ep.ch_names

    def extract_features(self, data, roi_fn, ch_names, y):
        roi_channels = set(roi_fn(ch_names))
        cols = [ch_names.index(cha) for cha in roi_channels if cha in ch_names]
        return data[:, cols, :]

    def score(self, feats, y):
        dec = SlidingEstimator(build_pipeline("ridge"), scoring="r2", n_jobs=1)
        return cross_val_multiscore(dec, feats, y, cv=CV_REG).mean(0)


# ----- TF Sliding ------------
class TFSlidingAnalysis(BaseAnalysis):
    """
    Analysis pipeline that concatenates TF epochs and evaluate using a ridge regression sliding estimator
    Subclass of Base Analysis
    """

    name = "TF Sliding R2"

    def run(self):
        if not self.subjects:
            logger.warning(f"No subjects for {self.name}, skipping.")
            return

        for roi_name, roi_fn in self.rois.items():
            # The difference between the analysis working with erp data is in the run,
            # when it is divided in a loop depending on the bands as well, to see better results
            for band, (fmin, fmax) in BANDS.items():
                logger.info(f"{self.name} ROI={roi_name}, band={band}:")
                all_scores = []
                times = None
                for subj in self.subjects:
                    data, ts, y, ch_names = self.load_and_preprocess(subj, fmin, fmax)
                    if times is None:
                        times = ts
                    feats = self.extract_features(data, roi_fn, ch_names, y)
                    score = self.score(feats, y)
                    all_scores.append(score)
                arr = np.vstack(all_scores)
                self.test_and_plot(arr, times, f"{roi_name} {band}")

    def load_and_preprocess(self, subj, fmin, fmax):
        # Load the data and concatenate it in one array
        tfr_list = [self.handler.load_tfr(subj, t) for t in ("search", "memory")]
        if not tfr_list:
            raise RuntimeError(f"No TFR data for subj {subj}")
        all_data = np.concatenate([tfr.data for tfr in tfr_list], axis=0)
        all_meta = pd.concat([tfr.metadata for tfr in tfr_list], ignore_index=True)
        tfr_tmpl = tfr_list[0]
        all_tfr = EpochsTFRArray(
            data=all_data,
            info=tfr_tmpl.info,
            freqs=tfr_tmpl.freqs,
            times=tfr_tmpl.times,
        )
        all_tfr.metadata = all_meta

        # Process the data not only for the timeframe and baseline, but also based on the given band
        tfr = self.handler.preprocess(all_tfr)
        freq_index = np.where((tfr.freqs >= fmin) & (tfr.freqs <= fmax))[0]
        data = tfr.data[:, :, freq_index, :].mean(axis=2)
        return data, tfr.times, tfr.metadata["mem_err"].values, tfr.ch_names

    def extract_features(self, data, roi_fn, ch_names, y):
        roi_channels = set(roi_fn(ch_names))
        cols = [ch_names.index(cha) for cha in roi_channels if cha in ch_names]
        return data[:, cols, :]

    def score(self, feats, y):
        dec = SlidingEstimator(build_pipeline("ridge"), scoring="r2", n_jobs=1)
        return cross_val_multiscore(dec, feats, y, cv=CV_REG).mean(0)


# ----- CSP Static ------------
class CSPStaticAnalysis(BaseAnalysis):
    """
    Static CSP analysis pipeline that concatenates ERP epochs and evaluate using a ridge regression
    Subclass of Base Analysis
    """

    name = "CSP Static R2"

    def load_and_preprocess(self, subj):
        ep = mne.concatenate_epochs(
            [self.handler.load_erp(subj, t) for t in ("search", "memory")]
        )
        ep = self.handler.preprocess(ep)
        return ep.get_data(), ep.times, ep.metadata["mem_err"].values, ep.ch_names

    def extract_features(self, data, roi_fn, ch_names, y):
        roi_channels = set(roi_fn(ch_names))
        cols = [ch_names.index(cha) for cha in roi_channels if cha in ch_names]
        X = data[:, cols, :]
        csp = CSP(n_components=CSP_COMPONENTS, reg="ledoit_wolf", log=True)
        X_sp = csp.fit_transform(X, y)
        return X_sp.mean(axis=2) if X_sp.ndim == 3 else X_sp

    def score(self, feats, y):
        X = feats.mean(axis=2) if feats.ndim == 3 else feats
        return cross_val_score(build_pipeline("ridge"), X, y, cv=CV_REG, scoring="r2")

    def test_and_plot(self, arr, times, roi_name):
        # Run a cluster-based test
        n_folds = arr.shape[1]
        folds = np.arange(n_folds)
        t_obs, clusters, p_vals, _ = spatio_temporal_cluster_1samp_test(
            arr, n_permutations=200, threshold=0.0, tail=1
        )
        sig = np.zeros(n_folds, bool)
        for idxs, p in zip(clusters, p_vals):
            if p <= 0.05:
                sig[idxs[0]] = True

        # Setting the information for the plot
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
        logger.info(f"{self.name}/{roi_name} folds={n_folds} mean_R2={mean.mean():.3f}")

        create_directory(f"{OUTPUT_DIR}/{self.name.lower().replace(' ', '_')}")
        # Plot set up
        plt.figure()
        plt.bar(folds, mean, yerr=sem, capsize=4)
        # Put a red star a little above the bar for each significant fold
        plt.scatter(folds[sig], mean[sig] + sem[sig] + 0.01, color="r", marker="*")
        plt.xticks(folds)
        plt.xlabel("CV fold index")
        plt.ylabel("Mean R²")
        plt.title(f"{self.name} ({roi_name})")
        plt.savefig(
            os.path.join(
                OUTPUT_DIR, f"{self.name.lower().replace(' ', '_')}/{roi_name}.svg"
            ),
            dpi=PLOT_DPI,
        )
        plt.close()


# ----- SPoC Static ------------
class SPoCStaticAnalysis(BaseAnalysis):
    """
    Static SPoc analysis pipeline that concatenates ERP epochs and evaluate using a Random Forest regression
    Subclass of Base Analysis
    """

    name = "SPoC Static R2"

    def run(self):
        if not self.subjects:
            logger.warning(f"No subjects for {self.name}, skipping.")
            return

        for roi_name, roi_fn in self.rois.items():
            # The difference between the analysis working with erp data is in the run,
            # when it is divided in a loop depending on the bands as well, to see better results
            for band, (fmin, fmax) in BANDS.items():
                logger.info(f"{self.name} ROI={roi_name}, band={band}:")
                all_scores = []
                times = None
                for subj in self.subjects:
                    data, ts, y, ch_names = self.load_and_preprocess(subj, fmin, fmax)
                    if times is None:
                        times = ts
                    feats = self.extract_features(data, roi_fn, ch_names, y)
                    score = self.score(feats, y)
                    all_scores.append(score)
                arr = np.vstack(all_scores)
                self.test_and_plot(arr, times, f"{roi_name} {band}")

    def load_and_preprocess(self, subj, fmin, fmax):
        tfr_list = [self.handler.load_tfr(subj, t) for t in ("search", "memory")]
        if not tfr_list:
            raise RuntimeError(f"No TFR data for subj {subj}")
        all_data = np.concatenate([t.data for t in tfr_list], axis=0)
        all_meta = pd.concat([t.metadata for t in tfr_list], ignore_index=True)
        tfr_tmpl = tfr_list[0]
        all_tfr = EpochsTFRArray(
            data=all_data,
            info=tfr_tmpl.info,
            freqs=tfr_tmpl.freqs,
            times=tfr_tmpl.times,
        )
        all_tfr.metadata = all_meta
        tfr = self.handler.preprocess(all_tfr)
        freq_index = np.where((tfr.freqs >= fmin) & (tfr.freqs <= fmax))[0]
        data = tfr.data[:, :, freq_index, :].mean(axis=2)
        return data, tfr.times, tfr.metadata["mem_err"].values, tfr.ch_names

    def extract_features(self, data, roi_fn, ch_names, y):
        roi_channels = set(roi_fn(ch_names))
        cols = [ch_names.index(cha) for cha in roi_channels if cha in ch_names]
        X = data[:, cols, :]
        spoc = SPoC(n_components=SPOC_COMPONENTS, reg="ledoit_wolf", log=True)
        X_sp = spoc.fit_transform(X, y)
        return X_sp.mean(axis=2) if X_sp.ndim == 3 else X_sp

    def score(self, feats, y):
        return cross_val_score(build_pipeline("rf"), feats, y, cv=CV_REG, scoring="r2")

    def test_and_plot(self, arr, times, roi_name):
        # Run a cluster-based test
        n_folds = arr.shape[1]
        folds = np.arange(n_folds)
        t_obs, clusters, p_vals, _ = spatio_temporal_cluster_1samp_test(
            arr, n_permutations=200, threshold=0.0, tail=1
        )
        sig = np.zeros(n_folds, bool)
        for idxs, p in zip(clusters, p_vals):
            if p <= 0.05:
                sig[idxs[0]] = True

        # Setting the information for the plot
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
        logger.info(f"{self.name}/{roi_name} folds={n_folds} mean_R2={mean.mean():.3f}")
        
        create_directory(f"{OUTPUT_DIR}/{self.name.lower().replace(' ', '_')}")
        # Plot set up
        plt.figure()
        plt.bar(folds, mean, yerr=sem, capsize=4)
        # Put a red star a little above the bar for each significant fold
        plt.scatter(folds[sig], mean[sig] + sem[sig] + 0.01, color="r", marker="*")
        plt.xticks(folds)
        plt.xlabel("CV fold index")
        plt.ylabel("Mean R²")
        plt.title(f"{self.name} ({roi_name})")
        plt.savefig(
            os.path.join(
                OUTPUT_DIR, f"{self.name.lower().replace(' ', '_')}/{roi_name}.svg"
            ),
            dpi=PLOT_DPI,
        )
        plt.close()


# -----Main Script-------------
if __name__ == "__main__":
    """
    Main execution: load data and run 4 different analyses
    """
    create_directory(OUTPUT_DIR)

    # Get lists with subjects for each type of files
    files_erp = glob.glob(os.path.join(DATA_DIR, "erp_fixed*.mat"))
    subjects_erp = sorted(
        {
            re.match(r"erp_fixed(\d+)\.mat$", os.path.basename(f)).group(1)
            for f in files_erp
            if re.match(r"erp_fixed(\d+)\.mat$", os.path.basename(f))
        }
    )
    files_tf = glob.glob(os.path.join(DATA_DIR, "tf_fixed*.mat"))
    subjects_tf = sorted(
        {
            re.match(r"tf_fixed(\d+)\.mat$", os.path.basename(f)).group(1)
            for f in files_tf
            if re.match(r"tf_fixed(\d+)\.mat$", os.path.basename(f))
        }
    )

    # Create a data handler
    dh = DataHandler(BASELINE, TIME_WINDOW)
    # Run all analysis to compare results
    ERPSlidingAnalysis(subjects_erp, ROI, dh).run()
    TFSlidingAnalysis(subjects_tf, ROI, dh).run()
    CSPStaticAnalysis(subjects_erp, ROI, dh).run()
    SPoCStaticAnalysis(subjects_tf, ROI, dh).run()

    logger.info("Done with this script")
