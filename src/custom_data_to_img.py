"""This script converts custom dataset samples into images for further processing.

Spectrograms are clipped to a percentile-based value range (see
`compute_clipping_values`) before being scaled to 8-bit PNGs, so that the
limited 8-bit resolution is not dominated by rare outlier values. In addition,
it generates a labels file containing the jammer type, and the number of
legitimate transmitters for each sample. Also, a metadata file containing the
clipping bounds of the spectrograms is generated to enable rescaling of the
spectrograms back to their original values.
"""

__docformat__ = "numpy"

import os
import sys

import argparse
import pickle as pkl
import warnings
from glob import glob
import hydra
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# import own modules
_repo_name = "ofdma-spectrum-anomalies-simulation"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]
sys.path.append(os.path.abspath(_module_path))

from src.utils.ofdm_utils import get_total_allocated_resources
from src.utils.data_utils import (
    get_datapath,
    get_spectrogram_img_filenamename,
    get_resource_alloc_img_filenamename,
    generate_dataset_splits,
)

# initialize the datasets path
_datapath = get_datapath(_repo_name)

# metrics that are stored per SU in the samples and that are written to one
# column per SU in the labels file
PER_SU_METRICS = [
    "snr_by_su",
    "sjr_by_su",
    "jsnr_local_by_su",
    "db_contrast_global_by_su",
    "db_contrast_local_by_su",
]

# Percentiles used to derive the clipping range for the 8-bit PNG conversion.
# The lower bound is taken from the distribution of all spectrogram values
# (jammed and non-jammed), the upper bound only from jammed samples, since
# jammer power is the anomaly-relevant signal and must not be clipped away
# too aggressively. Values were chosen based on the analysis in
# `notebooks/analyze_clipping_percentiles.ipynb`.
CLIP_LOW_PERCENTILE = 0.1
CLIP_HIGH_PERCENTILE_JAMMED = 99.99
CLIP_HISTOGRAM_BINS = 4000

# Generously wide value range assumed to fix the histogram bin edges up
# front, so the value histograms can be built in a single pass over the
# sample files instead of requiring one pass to find the exact min/max
# before a second pass to bin the values. It does not need to be exact - it
# only needs to safely contain all realistic spectrogram values; a runtime
# warning is raised in `compute_clipping_values` if the actual data exceeds
# it, since values outside this range are silently excluded from the
# histogram and could then bias the estimated percentiles.
ASSUMED_VALUE_RANGE_DB = (-260.0, 80.0)


def _weighted_percentile(bin_edges, counts, percentile):
    """Compute a percentile from a histogram via linear interpolation of the CDF.

    Parameters
    ----------
    bin_edges : np.ndarray
        Edges of the histogram bins, as returned by `np.histogram`.
    counts : np.ndarray
        Counts per bin, as returned by `np.histogram`.
    percentile : float
        Percentile to compute, in [0, 100].

    Returns
    -------
    float
        The value at the given percentile.
    """

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cum_counts = np.cumsum(counts)
    target = percentile / 100.0 * cum_counts[-1]
    return float(np.interp(target, cum_counts, bin_centers))


def compute_clipping_values(
    filenames,
    low_percentile=CLIP_LOW_PERCENTILE,
    high_percentile_jammed=CLIP_HIGH_PERCENTILE_JAMMED,
    num_bins=CLIP_HISTOGRAM_BINS,
    assumed_value_range=ASSUMED_VALUE_RANGE_DB,
):
    """Compute the clipping range used to scale spectrograms into 8-bit PNGs.

    The dataset-wide value range (min/max across all samples) is much wider
    than what most samples actually use, since a small fraction of extreme
    outlier values dominate it. Clipping to a percentile-based range before
    the min-max normalization therefore recovers substantially more of the
    8-bit resolution for the value range that samples actually occupy.

    The lower clip bound is computed from the distribution of all spectrogram
    values, since low values are dominated by the noise floor and carry no
    anomaly-relevant information. The upper clip bound is computed only from
    samples that contain a jammer, since jammer power is exactly the signal
    an anomaly detector needs to see, and clipping it based on the full
    (mostly non-jammed) distribution would cut it too aggressively.

    Both percentiles are estimated from histograms built in a single pass
    over the sample files (using `assumed_value_range` for the bin edges),
    so memory usage stays independent of the dataset size and the sample
    files only need to be read once here (matching the one read for image
    generation afterwards).

    Parameters
    ----------
    filenames : list
        List of filenames containing the samples.
    low_percentile : float
        Percentile (over all samples) used for the lower clip bound.
    high_percentile_jammed : float
        Percentile (over jammed samples only) used for the upper clip bound.
    num_bins : int
        Number of histogram bins used to estimate the percentiles.
    assumed_value_range : tuple of float
        (min, max) used to fix the histogram bin edges up front. Must safely
        contain the actual data range; violations are reported as a warning.

    Returns
    -------
    clip_min : float
        Lower clipping bound.
    clip_max : float
        Upper clipping bound.
    """

    bin_edges = np.linspace(assumed_value_range[0], assumed_value_range[1], num_bins + 1)
    hist_all = np.zeros(num_bins, dtype=np.int64)
    hist_jammed = np.zeros(num_bins, dtype=np.int64)

    min_val = np.inf
    max_val = -np.inf

    p_bar = tqdm(total=len(filenames), desc="Building value histograms")
    for filename in filenames:
        with open(filename, "rb") as f:
            samples = pkl.load(f)
        for sample in samples:
            is_jammed = len(sample.jammers) > 0
            for su_idx in sample.spectrograms:
                values = sample.spectrograms[su_idx]
                min_val = min(min_val, np.min(values))
                max_val = max(max_val, np.max(values))
                hist, _ = np.histogram(values.ravel(), bins=bin_edges)
                hist_all += hist
                if is_jammed:
                    hist_jammed += hist
        del samples  # free memory
        p_bar.update(1)

    p_bar.close()

    if min_val < assumed_value_range[0] or max_val > assumed_value_range[1]:
        warnings.warn(
            f"Spectrogram values ({min_val} to {max_val}) exceed the assumed "
            f"histogram range {assumed_value_range} (ASSUMED_VALUE_RANGE_DB). "
            "Values outside this range were excluded from the percentile "
            "estimate, which may bias the resulting clip bounds. Consider "
            "widening ASSUMED_VALUE_RANGE_DB."
        )

    clip_min = _weighted_percentile(bin_edges, hist_all, low_percentile)
    clip_max = _weighted_percentile(bin_edges, hist_jammed, high_percentile_jammed)

    print(f"Value range: {min_val} to {max_val}")
    print(f"Clip min (p{low_percentile} of all values): {clip_min}")
    print(f"Clip max (p{high_percentile_jammed} of jammed values): {clip_max}")

    return clip_min, clip_max


def flatten_per_su_dicts(labels):
    """Converts the per SU nested dicts into separate list columns.

    Transforms nested dicts like {"su_0": [val, val], "su_1": [val, val]} into
    separate keys like "snr_by_su_su_0", "snr_by_su_su_1", etc., enabling
    conversion to CSV format. This is done for all metrics listed in
    PER_SU_METRICS.

    Parameters
    ----------
    labels : dict
        Dictionary containing labels with the per SU metrics as nested dicts.

    Returns
    -------
    labels : dict
        Dictionary with flattened per SU columns and original dicts removed.
    """

    for metric_name in PER_SU_METRICS:
        if metric_name in labels:
            metric_dict = labels.pop(metric_name)
            for su_key, su_values in metric_dict.items():
                labels[f"{metric_name}_{su_key}"] = su_values

    return labels


def samples_to_imgs_and_labels(
    data_path,
    dataset_nr,
    filename,
    total_sample_idx,
    labels,
    clip_min,
    clip_max,
    binary_resource_img=True,
):
    """Creates images for easier processing of the dataset from the custom data format.

    Parameters
    -----
    data_path : str
        Path to the datasets folder.
    dataset_nr : int
        Dataset number.
    filename : str
        Filename of the samples in custom format (to process).
    total_sample_idx : int
        Index of the current sample.
    labels : dict
        Dictionary containing the labels (jammer type and number
        of legitimate transmitters).
    clip_min : float
        Lower clipping bound applied before scaling to 8-bit (for consistent scaling).
    clip_max : float
        Upper clipping bound applied before scaling to 8-bit (for consistent scaling).
    binary_resource_img : bool
        If True, the resource allocation image will be binary.
            If False, the resource allocation image will contain the corresponding
            transmitter index (starting with 1) for allocated resources, and 0 for non-allocated.
            Default: True.

    Returns
    -------
    total_sample_idx : int
        Updated index of the current sample.
    labels : dict
        Updated dictionary containing the labels (jammer type,
        jammer power, number of legitimate transmitters).
        Only updated for PT type samples! For DT type samples, the
        it is returned as is.
    """

    filename = os.path.join(
        data_path,
        f"{dataset_nr}",
        "custom",
        filename,
    )

    with open(filename, "rb") as f:
        samples = pkl.load(f)

    target_path = os.path.join(data_path, f"{dataset_nr}", "images")

    for sample in samples:
        # get label (jammer type)
        if len(sample.jammers) > 0:
            labels["jammer_type"].append(sample.jammers[0].type)
            labels["jammer_power"].append(sample.jammers[0].transmit_power)
            labels["jammer_location"].append(sample.jammers[0].location)
        else:
            labels["jammer_type"].append("no jammer")
            labels["jammer_power"].append(np.nan)
            labels["jammer_location"].append(np.nan)
        labels["num_legitimate_transmitters"].append(len(sample.transmitters))
        labels["jammer_occupancy"].append(getattr(sample, "jammer_occupancy", np.nan))

        # Collect the per SU metrics. The SU indexes are taken from the
        # spectrograms, so that a metric that is missing for a sample results in
        # NaN instead of in columns of inconsistent length.
        for metric_name in PER_SU_METRICS:
            metric_by_su = getattr(sample, metric_name, {})
            for su_idx in sample.spectrograms:
                if su_idx not in labels[metric_name]:
                    labels[metric_name][su_idx] = []
                labels[metric_name][su_idx].append(metric_by_su.get(su_idx, np.nan))

        # get total allocated resources (for all legitimate TX) and save as image
        total_allocated_resources = get_total_allocated_resources(
            sample, 12, 14, binary_resource_img
        ).astype(np.uint8)

        resource_img = Image.fromarray(total_allocated_resources)
        resource_img.save(
            os.path.join(
                target_path,
                get_resource_alloc_img_filenamename(total_sample_idx),
            )
        )

        # save the spectrograms as image
        for su_idx in sample.spectrograms:
            su_spectrogram = sample.spectrograms[su_idx]
            # clip first to avoid over-/underflow when casting to uint8, then
            # scale the clipped range to the 8-bit range for PNG output
            su_spectrogram = np.clip(su_spectrogram, clip_min, clip_max)
            su_spectrogram = (
                (su_spectrogram - clip_min) / (clip_max - clip_min) * 255
            ).astype(np.uint8)
            spectrogram_img = Image.fromarray(su_spectrogram)
            spectrogram_img.save(
                os.path.join(
                    target_path,
                    get_spectrogram_img_filenamename(total_sample_idx, su_idx),
                )
            )

        total_sample_idx += 1

    return total_sample_idx, labels


def parse_arguments(default_dataset_nr):
    """Parse command line arguments.

    Parameters
    ----------
    default_dataset_nr : int
        Default dataset number to use if not overridden via command line
        (taken from the dataset_generation.yaml config file).

    Returns
    -------
    args : Namespace
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate a dataset for a given scene and configuration."
    )

    parser.add_argument(
        "-d",
        "--dataset-number",
        default=default_dataset_nr,
        type=int,
        required=False,
        help="Dataset number to process.",
    )

    parser.add_argument(
        "--binary-resource-img",
        type=bool,
        default=False,
        required=False,
        help="If True, the resource allocation image will be binary. If False,"
        " the resource allocation image will contain the corresponding transmitter"
        " index (starting with 1) for allocated resources, and 0 for non-allocated.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # load config file which contains the parameters into cfg object
    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="dataset_generation")

    args = parse_arguments(cfg.dataset_nr)

    dataset_nr = int(args.dataset_number)
    print(f"Processing dataset number: {dataset_nr}")

    total_sample_idx = 0
    labels = {
        "jammer_type": [],
        "jammer_power": [],
        "jammer_location": [],
        "num_legitimate_transmitters": [],
        "jammer_occupancy": [],
    }
    labels.update({metric_name: {} for metric_name in PER_SU_METRICS})

    custom_files_dir = os.path.join(_datapath, f"{dataset_nr}", "custom")
    filenames = glob(
        os.path.join(
            custom_files_dir,
            f"samples-*.pkl",
        )
    )

    target_path = os.path.join(_datapath, f"{dataset_nr}", "images")

    # create folder for images or clear files the folder
    if os.path.exists(target_path):
        for file in os.listdir(target_path):
            os.remove(os.path.join(target_path, file))
    else:
        os.makedirs(target_path)

    clip_min, clip_max = compute_clipping_values(filenames)

    # Save the clipping bounds (columns are still named min_val/max_val for
    # backwards compatibility with existing readers of this file, e.g.
    # `load_dataset` in `src/utils/data_utils.py`) so that the clipped/rescaled
    # values can be restored. See the README for details on how these bounds
    # are derived.
    pd.DataFrame({"min_val": [clip_min], "max_val": [clip_max]}).to_csv(
        os.path.join(_datapath, f"{dataset_nr}", "spectrogram_min_max.csv"),
        index=False,
    )

    for file_idx, filename in enumerate(tqdm(filenames)):

        total_sample_idx, labels = samples_to_imgs_and_labels(
            _datapath,
            dataset_nr,
            filename,
            total_sample_idx,
            labels,
            clip_min,
            clip_max,
            binary_resource_img=args.binary_resource_img,
        )

    # Flatten the nested per SU dicts into separate columns
    labels = flatten_per_su_dicts(labels)

    # Assign reproducible train/valid/test splits for the supervised and
    # unsupervised protocols (see README for details).
    split_supervised, split_unsupervised = generate_dataset_splits(
        np.asarray(labels["jammer_type"]),
        train_frac=cfg.split.train_frac,
        test_frac=cfg.split.test_frac,
        valid_frac=cfg.split.valid_frac,
        left_out_types=tuple(cfg.split.left_out_types),
        seed=cfg.split.seed,
    )
    labels["split_supervised"] = split_supervised
    labels["split_unsupervised"] = split_unsupervised

    pd.DataFrame(labels).to_csv(
        os.path.join(_datapath, f"{dataset_nr}", "labels.csv"),
        index=False,
    )
