"""Utilities for loading and processing data."""

__docformat__ = "numpy"

import os

from glob import glob
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from PIL import Image
import warnings

# import tqdm either for script or for notebook depending on environment
try:
    # Check if we're in a Jupyter notebook environment
    from IPython import get_ipython

    if "IPKernelApp" in get_ipython().config:
        from tqdm.notebook import trange
    else:
        from tqdm import trange
except Exception:
    # Fall back to regular tqdm
    from tqdm import trange


def get_datapath(repo_name: str):
    """Get the path to the dataset directory.

    Parameters
    ----------
    repo_name : str
        The name of the repository, e.g., 'ofdma-spectrum-anomalies-simulation'

    Returns
    -------
    datapath: str
        The path to the dataset directory.
    """
    module_path = __file__[: __file__.find(repo_name) + len(repo_name)]
    with open(os.path.join(module_path, "datapath.txt"), "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.strip().startswith("#") or line.strip().startswith("<"):
            continue

        if len(line.strip()) > 1:
            if os.path.isdir(line.strip()):
                return line.strip()
            else:
                print(
                    f"Line {idx + 1} in datapath.txt does not point to a valid directory: {line.strip()}"
                )

    raise ValueError(
        "No valid datapath found in the datapath.txt file. "
        "Please check the file and ensure it contains a valid path."
    )


def get_spectrogram_img_filenamename(sample_idx: int, su_idx: int):
    """Generate the filename for a spectrogram image based on the mode, sample index and SU index.

    Parameters
    ----------
    sample_idx : int
        The index of the sample.
    su_idx : int / str
        The index of the SU (secondary user) in the sample.
        Instead of the number, a wildcard '*' can be used to create a regex pattern.

    Returns
    -------
    filename: str
        The filename for the spectrogram image.
    """

    if su_idx != "*":
        su_idx = str(su_idx).zfill(2)

    return f"spectrogram-{str(sample_idx).zfill(5)}-{su_idx}.png"


def get_resource_alloc_img_filenamename(sample_idx: int):
    """Generate the filename for the resource allocation image based on the sample index.

    Parameters
    ----------
    sample_idx : int
        The index of the sample.

    Returns
    -------
    filename: str
        The filename for the resource allocation image.
    """

    if sample_idx != "*":
        sample_idx = str(sample_idx).zfill(5)

    return f"alloc_res-{sample_idx}.png"


def load_allocations(
    dataset_path: str, num_freqbins_aggr: int, num_timeslots_aggr: int
):
    """Load the resource allocation images from the dataset.

    Returns
    -------
    allocations: np.ndarray
        The resource allocations as a numpy array with shape
        (num_freq_bins, num_time_bins).
    """

    img_path = os.path.join(dataset_path, "images")

    file_list = glob(os.path.join(img_path, get_resource_alloc_img_filenamename("*")))

    for sample_idx, file_name in enumerate(file_list):

        img = np.array(Image.open(file_name))

        if sample_idx == 0:
            allocations = np.zeros((len(file_list), *img.shape))

        allocations[sample_idx] = img

    # Aggregate the allocations if needed
    if num_freqbins_aggr > 1 or num_timeslots_aggr > 1:
        allocations = aggregate_spectrograms(
            allocations, num_freqbins_aggr, num_timeslots_aggr
        )

    return allocations


def load_one_sample_from_images(
    img_path: str, mode: str, sample_idx: int, aggregation: str = "mean"
) -> np.ndarray:
    """Load a single sample from the images.

    Parameters
    ----------
    mode : str
        The mode of the sample, i.e., 'PT' or 'DT'.
    sample_idx : int
        The index of the sample.
    aggregation : str
        The aggregation method to use for the spectrograms.
        Currently only 'mean' is supported.

    Returns
    -------
    spectrograms : np.ndarray
        The aggregated spectrograms of the sample as a numpy array.
    """

    if mode not in ["PT", "DT"]:
        raise ValueError(f"Mode {mode} is not supported. Use 'PT' or 'DT'.")

    if aggregation not in ["mean", "none"]:
        raise NotImplementedError(
            f"Aggregation method {aggregation} is not implemented. Only 'mean' or None are supported."
        )

    file_list = glob(
        os.path.join(img_path, get_spectrogram_img_filenamename(mode, sample_idx, "*"))
    )

    for su_idx, filename in enumerate(file_list):

        img = np.array(Image.open(filename))

        if su_idx == 0:
            spectrograms = np.zeros((len(file_list), *img.shape), dtype=np.float32)

        spectrograms[su_idx] = img

    if aggregation == "mean":
        spectrograms = np.mean(spectrograms, axis=0)

    return spectrograms


def aggregate_spectrograms(
    spectrograms: np.ndarray,
    num_freqbins_aggr: int,
    num_timebins_aggr: int,
    convert_from_log_to_lin_before: bool = False,
):
    """Aggregate the spectrograms to the given frequency and time bins.

    Parameters
    ----------
    spectrograms : np.ndarray
        The spectrograms to aggregate.
        Shape: (num_samples, num_freq_bins, num_time_bins).
    num_freq_bins_aggr : int
        The number of frequency bins which are aggregated.
    num_time_bins_aggr : int
        The number of time bins which are aggregated.
    convert_from_log_to_lin_before : bool
        Whether to convert the spectrograms from the logarithmic domain (dB) to the linear domain before aggregation.
        Default is False, which means no conversion is done.

    Returns
    -------
    np.ndarray
        The aggregated spectrograms.
        Shape: (num_samples, num_freq_bins // num_freq_bins_to_aggregate,
                num_time_bins // num_time_bins_to_aggregate).
    """

    if len(spectrograms.shape) not in [3, 4]:
        raise ValueError(
            "Spectrograms must be a 3D array with shape (num_samples, num_freq_bins, num_time_bins) or"
            " a 4D array with shape (num_samples, num_sus, num_freq_bins, num_time_bins)."
        )

    if convert_from_log_to_lin_before:
        spectrograms = np.power(10, spectrograms / 10)  # convert dB to linear scale

    if num_freqbins_aggr > 1:
        freq_axis = -2  # Second to last axis (axis 1 for 3D, axis 2 for 4D)
        shape = list(spectrograms.shape)
        new_shape = (
            shape[:freq_axis]
            + [shape[freq_axis] // num_freqbins_aggr, num_freqbins_aggr]
            + shape[freq_axis + 1 :]
        )
        spectrograms = spectrograms.reshape(new_shape).mean(axis=freq_axis)

    if num_timebins_aggr > 1:

        time_axis = -1  # Last axis (axis 2 for 3D, axis 3 for 4D)
        if spectrograms.shape[time_axis] % num_timebins_aggr != 0:
            warnings.warn(
                f"Spectrograms time dimension {spectrograms.shape[time_axis]}"
                f" is not divisible by {num_timebins_aggr}, "
                f"will be truncated."
            )
            spectrograms = spectrograms[
                :, :, : -(spectrograms.shape[time_axis] % num_timebins_aggr)
            ]

        spectrograms = np.power(10, spectrograms / 10)  # convert dB to linear scale
        shape = list(spectrograms.shape)
        new_shape = shape[:time_axis] + [
            shape[time_axis] // num_timebins_aggr,
            num_timebins_aggr,
        ]
        spectrograms = spectrograms.reshape(new_shape).mean(axis=time_axis)

    if convert_from_log_to_lin_before:
        spectrograms = 10 * np.log10(spectrograms)  # convert back to dB

    return spectrograms


def load_dataset(
    datapath: str,
    aggregation_su: str,
    num_freqbins_aggr: int,
    num_timeslots_aggr: int,
    load_dt_spectrograms: bool = True,
    verbose: int = 0,
    batch_size: int = 100,
) -> tuple:
    """Load the dataset from the given path.

    Parameters
    ----------
    datapath : str
        The path to the dataset directory.
    aggregation_su : str
        The aggregation method to use to combine the spectrograms of the different
        sensing units. Aggregation is done in the logarithmic domain.
    num_freqbins_aggr : int
        The number of frequency bins to aggregate. Aggregation is done by averaging
        in the linear domain.
    num_timeslots_aggr : int
        The number of time bins to aggregate. Aggregation is done by averaging
        in the linear domain.
    load_dt_spectrograms : bool
        Whether to load and return the DT spectrograms.
    verbose : int
        The verbosity level for progress report. Default is 0.
    batch_size : int
        The number of samples to load in each parallel batch. Default is 100.

    Returns
    -------
    labels_df : pd.DataFrame
        The labels of the dataset as a pandas DataFrame.
    pt_spectrograms : np.ndarray
        The spectrograms of the PT samples as a numpy array with
        shape (num_samples, num_freq_bins, num_time_bins).
    dt_spectrograms : np.ndarray
        The spectrograms of the DT samples as a numpy array with
        shape (num_samples, num_freq_bins, num_time_bins).
        Only returned if `load_dt_spectrograms` is True.
    """

    img_path = os.path.join(datapath, "images")

    # load the labels
    labels_df = pd.read_csv(os.path.join(datapath, "labels.csv"))

    # load minimum and maximum values for scaling
    min_max_df = pd.read_csv(os.path.join(datapath, "spectrogram_min_max.csv"))
    v_min = min_max_df["min_val"].values[0]
    v_max = min_max_df["max_val"].values[0]

    if verbose > 0:
        print(f"\nSpectrogram value range: {v_min} to {v_max}")
        print(f"Loading spectrograms...", end=" ")

    def load_spectrograms_batch(mode, start_idx, end_idx):
        """Load a batch of spectrograms for the given mode and sample range."""
        # Get shape from the first sample in the batch
        first_spectrogram = load_one_sample_from_images(
            img_path, mode, start_idx, aggregation_su
        )

        batch_size_actual = end_idx - start_idx
        batch_spectrograms = np.zeros(
            (batch_size_actual, *first_spectrogram.shape), dtype=np.float32
        )
        batch_spectrograms[0] = first_spectrogram

        for i, sample_idx in enumerate(range(start_idx + 1, end_idx)):
            spectrogram = load_one_sample_from_images(
                img_path, mode, sample_idx, aggregation_su
            )
            batch_spectrograms[i + 1] = spectrogram

        batch_spectrograms = aggregate_spectrograms(
            batch_spectrograms,
            num_freqbins_aggr,
            num_timeslots_aggr,
            convert_from_log_to_lin_before=True,
        )

        # rescale to the original value range
        batch_spectrograms = v_min + batch_spectrograms / 255 * (v_max - v_min)

        return batch_spectrograms

    def load_spectrograms(mode):
        """Load all spectrograms for a given mode using parallel batches."""
        num_samples = labels_df.shape[0]

        # Create batch indices
        batch_starts = list(range(0, num_samples, batch_size))
        batch_ends = [min(start + batch_size, num_samples) for start in batch_starts]

        if verbose > 0:
            print(
                f"Loading {mode} spectrograms in {len(batch_starts)} batches...",
                end=" ",
            )

        # Load batches in parallel
        batch_results = Parallel(verbose=int(10 * verbose), n_jobs=4)(
            delayed(load_spectrograms_batch)(mode, start, end)
            for start, end in zip(batch_starts, batch_ends)
        )

        # Concatenate all batches
        spectrograms = np.concatenate(batch_results, axis=0)

        if verbose > 0:
            print("DONE.")

        return spectrograms

    # Load PT spectrograms
    pt_spectrograms = load_spectrograms("PT")

    # Load DT spectrograms if requested
    if load_dt_spectrograms:
        dt_spectrograms = load_spectrograms("DT")

    if verbose > 0:
        print("done.")

    if load_dt_spectrograms:
        return labels_df, pt_spectrograms, dt_spectrograms
    else:
        return labels_df, pt_spectrograms


def generate_dataset_splits(
    jammer_types,
    supervised,
    unsupervised,
    left_out_types=("random_hop",),
):
    """Generate fixed-count train/valid/test split columns for the dataset.

    Reproduces the split used for the paper submission: for each protocol, a
    fixed number of samples per class is taken positionally from the
    (generation-order) sample sequence -- the first `num_train_*` samples of
    a class go to train, the next `num_valid_*` to valid, and the *last*
    `num_test_*` samples of the class go to test. Anomaly counts are divided
    evenly across jammer types, with `left_out_types` (e.g. `random_hop`)
    held out of training/validation entirely for out-of-distribution
    generalization testing. See the "Data Splits" section in the README for
    the full rationale.

    Parameters
    ----------
    jammer_types : np.array
        The jammer type label (e.g. "no jammer", "barrage", ...) of each
        sample, in the order the split columns should be returned in. Must
        be in generation order, since the split is positional.
    supervised : Mapping
        `num_train_normal`, `num_valid_normal`, `num_test_normal`,
        `num_train_anomaly`, `num_valid_anomaly`, `num_test_anomaly`. The
        anomaly counts are divided evenly across the jammer types not in
        `left_out_types` (train/valid) resp. across every jammer type (test).
    unsupervised : Mapping
        `num_train_normal`, `num_valid_normal`, `num_test_normal`,
        `num_test_anomaly` (unsupervised never trains/validates on anomaly
        samples). The test anomaly count is divided evenly across every
        jammer type.
    left_out_types : tuple of str
        Jammer types that are only ever assigned to the test split (for
        supervised learning), so that they can be used to evaluate
        generalization to a jammer type unseen during training. They are
        never assigned to train/valid in either protocol. Default: `("random_hop",)`.

    Returns
    -------
    split_supervised : np.array
        Split assignment for each sample for the supervised protocol.
        Values are `"train"`, `"valid"`, `"test"`, or `None` (unused).
    split_unsupervised : np.array
        Split assignment for each sample for the unsupervised protocol.
        Values are `"train"`, `"valid"`, `"test"`, or `None` (unused).
    """

    jammer_types = np.asarray(jammer_types)
    split_supervised = np.full(len(jammer_types), None, dtype=object)
    split_unsupervised = np.full(len(jammer_types), None, dtype=object)

    anomaly_types = [t for t in np.unique(jammer_types) if t != "no jammer"]
    train_types = [t for t in anomaly_types if t not in left_out_types]

    # "no jammer": positional split, independent per protocol
    for split_arr, cfg in ((split_supervised, supervised), (split_unsupervised, unsupervised)):
        idxs = np.where(jammer_types == "no jammer")[0]
        n_train, n_valid, n_test = (
            cfg["num_train_normal"],
            cfg["num_valid_normal"],
            cfg["num_test_normal"],
        )
        if n_train + n_valid + n_test > len(idxs):
            raise ValueError(
                f"Not enough 'no jammer' samples ({len(idxs)}) to satisfy"
                f" num_train_normal={n_train}, num_valid_normal={n_valid},"
                f" num_test_normal={n_test}."
            )
        split_arr[idxs[:n_train]] = "train"
        split_arr[idxs[n_train : n_train + n_valid]] = "valid"
        split_arr[idxs[-n_test:]] = "test"

    # anomaly types: train/valid counts divided evenly across train_types
    # (supervised only), test counts divided evenly across all anomaly_types
    per_train = supervised["num_train_anomaly"] // len(train_types)
    per_valid = supervised["num_valid_anomaly"] // len(train_types)
    per_test_supervised = supervised["num_test_anomaly"] // len(anomaly_types)
    per_test_unsupervised = unsupervised["num_test_anomaly"] // len(anomaly_types)

    for jammer_type in anomaly_types:
        idxs = np.where(jammer_types == jammer_type)[0]
        is_train_type = jammer_type in train_types

        needed_supervised = (per_train + per_valid if is_train_type else 0) + per_test_supervised
        if needed_supervised > len(idxs):
            raise ValueError(
                f"Not enough '{jammer_type}' samples ({len(idxs)}) to satisfy"
                f" the supervised per-type train/valid/test counts"
                f" ({per_train}/{per_valid}/{per_test_supervised})."
            )
        if per_test_unsupervised > len(idxs):
            raise ValueError(
                f"Not enough '{jammer_type}' samples ({len(idxs)}) to satisfy"
                f" the unsupervised per-type test count ({per_test_unsupervised})."
            )

        if is_train_type:
            split_supervised[idxs[:per_train]] = "train"
            split_supervised[idxs[per_train : per_train + per_valid]] = "valid"

        # test split taken from the end, independently per protocol
        split_supervised[idxs[-per_test_supervised:]] = "test"
        split_unsupervised[idxs[-per_test_unsupervised:]] = "test"

    return split_supervised, split_unsupervised


def downsample_spectrograms(
    spectrograms: np.ndarray,
    num_freq_bins_to_aggregate: int,
    num_timesteps_to_aggregate: int,
) -> np.ndarray:
    """Downsample the spectrograms to the given frequency and time bins.

    Parameters
    ----------
    spectrograms: np.ndarray
        The spectrograms to downsample.
        Shape: (num_samples, num_sus, num_freq_bins, num_time_bins).
    num_freq_bins_to_aggregate: int
        The number of frequency bins to aggregate.
    num_timesteps_to_aggregate: int
        The number of time bins to aggregate.

    Returns
    -------
    np.ndarray
        The downsampled spectrograms.
        Shape: (num_samples, num_sus, num_freq_bins // num_freq_bins_to_aggregate,
                num_time_bins // num_timesteps_to_aggregate).
    """

    reshaped = spectrograms.reshape(
        spectrograms.shape[0],
        spectrograms.shape[1],
        spectrograms.shape[2] // num_freq_bins_to_aggregate,
        num_freq_bins_to_aggregate,
        spectrograms.shape[3] // num_timesteps_to_aggregate,
        num_timesteps_to_aggregate,
    )

    downsampled = reshaped.mean(axis=(3, 5))

    return downsampled


def jammer_types_to_binary(jammer_labels: np.ndarray) -> np.ndarray:
    """Transform the jammer types to binary labels.

    Parameters
    ----------
    jammer_labels: np.array
        The jammer types (provided as a strings).
        No jammer present shall be indicated by 'None'.
        Every other type is interpreted as anomaly.

    Returns
    -------
    np.array
        The binary labels. Anomaly is indicated by True, normal by False.
    """

    return jammer_labels != "None"


if __name__ == "__main__":

    dataset_idx = 0

    datapath = get_datapath()
    labels_df = pd.read_csv(os.path.join(datapath, str(dataset_idx), "labels.csv"))

    load_one_sample_from_images(
        os.path.join(datapath, str(dataset_idx), "images"), "PT", 0, "none"
    )
