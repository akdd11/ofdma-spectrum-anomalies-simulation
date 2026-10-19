"""
Threshold-based baseline: energy detector with one threshold per time-frequency bin.

The input is the same as for the VAE and the ResNet: the spectrogram of every SU,
aggregated from subcarriers to resource blocks (RBs) on the frequency axis, i.e.,
num_rb x num_symbols bins (RB x OFDM symbol). For every SU and every bin, the power
threshold is a quantile of the received power of the normal training samples.
The anomaly score of a spectrogram is the largest exceedance of the threshold map of
its SU (in dB). The per-SU scores are z-normalized per SU on the normal validation
samples and averaged over the SUs, which gives one score per sample.

The evaluation follows the VAE in Notebooks/Evaluation.ipynb:
- the threshold map is fitted on the normal samples of split_unsupervised == "train",
  the per-SU normalization uses split_unsupervised == "valid",
- ROC curve and AUC per jammer type against the normal samples of
  split_unsupervised == "test".

Several quantiles can be evaluated in one run (the spectrograms are loaded once).
The scores of the validation and test samples are saved for every quantile, so that
ROC curves and threshold-based metrics (e.g., F1 with the threshold at a percentile
of the normal validation scores, as for the VAE) can be recomputed from scores.csv.

The dataset is <datasets folder from datapath.txt>/<dataset number>, as for the
scripts in <repo>/src.

Usage:
    python src/threshold_detector.py [-d <dataset number>] [--quantiles 0.5 0.9 0.95 0.99 0.999]
"""

# Import ---------------------------------------------------

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

# spectrogram loading utilities of the dataset generation in <repo>/src
_repo_name = "ofdma-spectrum-anomalies-simulation"
_example_use_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_repo_root = os.path.dirname(_example_use_path)
sys.path.insert(0, _repo_root)

from src.utils.data_utils import aggregate_spectrograms, get_datapath, load_one_sample_from_images


# Constants ---------------------------------------------------

NO_JAMMER = "no jammer"
JAMMER_TYPES = ["barrage", "deceptive", "pilot", "sweep", "random_hop"]
JAMMER_NAMES = {
    "barrage": "Barrage",
    "deceptive": "Deceptive",
    "pilot": "Pilot",
    "sweep": "Sweep",
    "random_hop": "Random hop",
}
NUM_SC_PER_RB = 12


# Data Loading ------------------------------------------------

def get_split(df):
    """
    Sample indices of the unsupervised split (identical to the slicing of
    train_unsupervised.py and Notebooks/Evaluation.ipynb with config.yml).
    """
    if "split_unsupervised" not in df.columns:
        raise KeyError(
            "labels.csv has no 'split_unsupervised' column. Re-export the dataset "
            "with src/custom_data_export.py to add the split columns."
        )

    split = {name: df.index[df["split_unsupervised"] == name].to_numpy() for name in ["train", "valid", "test"]}
    for name in ["train", "valid"]:
        assert (df.loc[split[name], "jammer_type"] == NO_JAMMER).all(), (
            f"Unsupervised {name} split contains anomalous samples"
        )
    return split


def load_spectrograms_db(data_path, sample_ids, n_jobs, desc="Loading"):
    """
    Load the spectrograms of the given samples in dB, aggregated to RBs like the VAE input.

    The power of the subcarriers of an RB is averaged in the linear domain, while the
    VAE input sums it. This is a constant offset of 10*log10(12) dB, which cancels in
    the exceedance of the threshold map.

    Returns
    -------
    np.ndarray, shape (num_samples, num_su, num_rb, num_symbols), dtype float32, in dB.
    """
    img_path = os.path.join(data_path, "images")
    min_max_df = pd.read_csv(os.path.join(data_path, "spectrogram_min_max.csv"))
    v_min = min_max_df["min_val"].values[0]
    v_max = min_max_df["max_val"].values[0]

    def load_sample(sample_idx):
        spectrograms = load_one_sample_from_images(img_path, sample_idx, aggregation="none")
        spectrograms = v_min + spectrograms / 255 * (v_max - v_min)
        spectrograms = aggregate_spectrograms(
            spectrograms[None], NUM_SC_PER_RB, 1, convert_from_log_to_lin_before=True
        )[0]
        return spectrograms.astype(np.float32)

    # processes, since load_one_sample_from_images globs the image folder for every sample
    frames = Parallel(n_jobs=n_jobs)(delayed(load_sample)(n) for n in tqdm(sample_ids, desc=desc))

    return np.stack(frames)


# Detector ------------------------------------------------

def fit_threshold_maps(X, quantiles):
    """
    X: normal training spectrograms in dB, shape (num_samples, num_su, num_rb, num_symbols)

    Returns
    -------
    np.ndarray, shape (len(quantiles), num_su, num_rb, num_symbols), in dB.
    """
    return np.quantile(X, quantiles, axis=0).astype(np.float32)


class BinwiseEnergyDetector:

    def __init__(self, threshold_map):
        self.threshold_map = threshold_map  # shape (num_su, num_rb, num_symbols), in dB
        self.su_mean = None                 # shape (num_su,)
        self.su_std = None                  # shape (num_su,)

    def score_per_su(self, X):
        """
        Largest exceedance of the threshold map in dB, shape (num_samples, num_su).
        """
        return np.max(X - self.threshold_map[None], axis=(-2, -1))

    def fit_normalization(self, X):
        """
        X: normal validation spectrograms, used for the per-SU z-normalization.
        """
        s = self.score_per_su(X)
        self.su_mean = s.mean(axis=0)
        self.su_std = s.std(axis=0)
        assert (self.su_std > 0).all(), "zero division"

    def score(self, X):
        """
        Anomaly score per sample: z-normalized per-SU scores averaged over the SUs.
        """
        return np.mean((self.score_per_su(X) - self.su_mean) / self.su_std, axis=1)


# Evaluation ----------------------------------------------

def auc_table(scores, jammer_types, jammer_powers):
    """
    AUC of every jammer type against all normal test samples, over all jammer
    powers ("all") and per jammer power, plus all jammer types pooled.
    """
    normal = jammer_types == NO_JAMMER
    rows = []
    for jammer in JAMMER_TYPES + ["all"]:
        anomalous = ~normal if jammer == "all" else jammer_types == jammer
        for power in ["all"] + sorted(np.unique(jammer_powers[anomalous])):
            sel = anomalous if power == "all" else anomalous & (jammer_powers == power)
            mask = normal | sel
            fpr, tpr, _ = roc_curve(sel[mask], scores[mask])
            rows.append(
                {
                    "jammer_type": jammer,
                    "jammer_power": power if power == "all" else f"{power:g}",
                    "num_anomalous": sel.sum(),
                    "auc": auc(fpr, tpr),
                }
            )
    return pd.DataFrame(rows)


def plot_roc(scores, jammer_types, filename):
    """
    ROC curve per jammer type against the normal test samples, in the style of
    the ROC curves of the VAE and ResNet in Notebooks/Evaluation.ipynb.
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 8
    plt.rcParams["pdf.fonttype"] = 42  # embed TrueType instead of Type 3 fonts

    normal = jammer_types == NO_JAMMER
    plt.figure(figsize=(4.6, 4.9))
    for jammer in JAMMER_TYPES:
        mask = normal | (jammer_types == jammer)
        fpr, tpr, _ = roc_curve(jammer_types[mask] == jammer, scores[mask])
        plt.plot(fpr, tpr, label=f"{JAMMER_NAMES[jammer]} ({auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="k", alpha=0.2)
    plt.grid(True)
    plt.xlabel("False Positive Rate", fontsize=17)
    plt.ylabel("True Positive Rate", fontsize=17)
    plt.subplots_adjust(left=0.16)

    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis="y", labelsize=17)

    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis="x", labelsize=17)

    plt.legend(fontsize=14, loc="lower right")
    # no creation date, so that regenerating an unchanged figure gives an identical file
    plt.savefig(filename, metadata={"CreationDate": None})
    plt.close()


# Main Routine --------------------------------------------

def score_column(quantile):
    return f"score_q{quantile:g}"


def main(args):
    data_path = os.path.join(get_datapath(_repo_name), str(args.dataset_number))
    out_dir = os.path.join(_example_use_path, args.out_dir)  # relative paths are relative to Example_Use
    print(f"Dataset: {data_path}")

    df = pd.read_csv(os.path.join(data_path, "labels.csv"), low_memory=False)
    split = get_split(df)
    print("Samples per split:", {k: len(v) for k, v in split.items()})

    X = load_spectrograms_db(data_path, split["train"], args.n_jobs, desc="Loading train")
    threshold_maps = fit_threshold_maps(X, args.quantiles)
    del X

    X_valid = load_spectrograms_db(data_path, split["valid"], args.n_jobs, desc="Loading valid")
    X_test = load_spectrograms_db(data_path, split["test"], args.n_jobs, desc="Loading test")

    scores = {"valid": {}, "test": {}}
    for quantile, threshold_map in zip(args.quantiles, threshold_maps):
        detector = BinwiseEnergyDetector(threshold_map)
        detector.fit_normalization(X_valid)
        scores["valid"][quantile] = detector.score(X_valid)
        scores["test"][quantile] = detector.score(X_test)
    del X_valid, X_test

    jammer_types = df.loc[split["test"], "jammer_type"].to_numpy()
    jammer_powers = df.loc[split["test"], "jammer_power"].to_numpy()

    # ------------------------------------------------
    # Evaluate & save
    # ------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    aucs = []
    for quantile in args.quantiles:
        a = auc_table(scores["test"][quantile], jammer_types, jammer_powers)
        a.insert(0, "quantile", quantile)
        aucs.append(a)
        plot_roc(scores["test"][quantile], jammer_types, os.path.join(out_dir, f"roc_q{quantile:g}.pdf"))
    aucs = pd.concat(aucs, ignore_index=True)
    aucs.to_csv(os.path.join(out_dir, "auc.csv"), index=False)

    # Scores of the normal validation samples (for a decision threshold) and of the test samples
    score_df = []
    for name in ["valid", "test"]:
        s = df.loc[split[name], ["jammer_type", "jammer_power"]].copy()
        s.insert(0, "sample", split[name])
        s.insert(1, "split", name)
        for quantile in args.quantiles:
            s[score_column(quantile)] = scores[name][quantile]
        score_df.append(s)
    pd.concat(score_df, ignore_index=True).to_csv(os.path.join(out_dir, "scores.csv"), index=False)

    np.savez_compressed(
        os.path.join(out_dir, "threshold_maps.npz"), quantiles=args.quantiles, threshold_maps=threshold_maps
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("\nAUC over all jammer powers:")
        overall = aucs[aucs["jammer_power"] == "all"]
        print(overall.pivot(index="quantile", columns="jammer_type", values="auc")[JAMMER_TYPES + ["all"]])
        print("\nAUC over all jammer types per jammer power:")
        pooled = aucs[aucs["jammer_type"] == "all"]
        table = pooled.pivot(index="quantile", columns="jammer_power", values="auc")
        print(table[list(dict.fromkeys(pooled["jammer_power"]))])
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    # default dataset number from the dataset generation config
    with open(os.path.join(_repo_root, "src", "conf", "dataset_generation.yaml")) as f:
        default_dataset_nr = yaml.safe_load(f)["dataset_nr"]

    parser = argparse.ArgumentParser(description="Energy detector baseline with one threshold per time-frequency bin")
    parser.add_argument(
        "-d", "--dataset_number", type=int, default=default_dataset_nr,
        help="Dataset number, i.e., subfolder of the datasets folder given in datapath.txt of the repository",
    )
    parser.add_argument(
        "--out_dir", type=str, default=os.path.join("Results", "threshold"),
        help="Output folder, relative to Example_Use; existing results are overwritten",
    )
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=[0.5, 0.9, 0.95, 0.99, 0.999],
        help="Quantiles of the normal training power per SU and bin used as threshold, one detector per quantile",
    )
    parser.add_argument("--n_jobs", type=int, default=32, help="Processes for loading the PNG files")
    main(parser.parse_args())
