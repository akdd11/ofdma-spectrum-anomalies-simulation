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


def generate_splitting_idxs(
    labels, num_train, num_valid, num_test, anomaly_probability_test
):
    """Generate indices for splitting the dataset into training,
    validation and test set.

    There are only normal samples in the training and validation set,
    since the splitting is intended for unsupervised learning.

    Parameters
    ----------
    labels : np.array
        The labels of the dataset.
    num_train : int
        The number of samples in the training set (only normal samples).
    num_valid : int
        The number of samples in the validation set (only normal samples).
    num_test : int
        The number of samples in the test set (normal and anomaly samples).
    anomaly_probability_test : float
        The probability of an anomaly in the test set.

    Returns
    -------
    train_idxs : np.array
        The indices of the training set.
    valid_idxs : np.array
        The indices of the validation set.
    test_idxs : np.array
        The indices of the test set.
    """

    if num_train + num_valid + num_test > len(labels):
        raise ValueError(
            "The number of samples in the splits exceeds the number"
            " of samples in the dataset."
        )

    if (
        num_train + num_valid + num_test * anomaly_probability_test
        > (labels == "no jammer").sum()
    ):
        raise ValueError(
            "The number of normal samples in the splits exceeds the"
            " number of normal samples in the dataset."
        )

    random_generator = np.random.default_rng(seed=42)

    jammer_idxs_by_type = {}
    for jammer_type in np.unique(labels):
        jammer_idxs_by_type[jammer_type] = np.where(labels == jammer_type)[0]
        random_generator.shuffle(jammer_idxs_by_type[jammer_type])

    train_idxs = jammer_idxs_by_type["no jammer"][:num_train]
    valid_idxs = jammer_idxs_by_type["no jammer"][num_train : num_train + num_valid]

    num_anomalies = int(num_test * anomaly_probability_test)
    num_anomalies_by_type = num_anomalies // (len(jammer_idxs_by_type) - 1)
    # number of anomalies for each jammer type, normal samples are not considered
    num_normals_test = num_test - num_anomalies

    test_idxs = jammer_idxs_by_type["no jammer"][
        num_train + num_valid : num_train + num_valid + num_normals_test
    ]
    if len(test_idxs) < num_normals_test:
        raise ValueError(
            "Not enough normal samples for the test set. "
            "Please adjust the number of samples in the splits."
        )

    for jammer_type in jammer_idxs_by_type:
        if jammer_type == "no jammer":
            continue
        test_idxs = np.concatenate(
            (test_idxs, jammer_idxs_by_type[jammer_type][:num_anomalies_by_type])
        )

    return train_idxs, valid_idxs, test_idxs


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

    # num_samples_total = labels_df.shape[0]

    # num_train = int(num_samples_total * 0.6)
    # num_val = int(num_samples_total * 0.0)
    # num_test = int(num_samples_total * 0.2)

    # anomaly_probability = 0.5

    # train_idxs, valid_idxs, test_idxs = generate_splitting_idxs(
    #     labels_df["jammer_type"].to_numpy(),
    #     num_train,
    #     num_val,
    #     num_test,
    #     anomaly_probability,
    # )

    # labels_test = labels_df["jammer_type"].to_numpy()[test_idxs]

    load_one_sample_from_images(
        os.path.join(datapath, str(dataset_idx), "images"), "PT", 0, "none"
    )
