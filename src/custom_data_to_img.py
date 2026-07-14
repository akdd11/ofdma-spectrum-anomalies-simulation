"""This script converts custom dataset samples into images for further processing.

In addition, it generates a labels file containing the jammer type, and the number of
legitimate transmitters for each sample. Also, a metadata file
containing the minimum and maximum values of the spectrograms is generated to enable
rescaling of the spectrograms back to their original values.
"""

__docformat__ = "numpy"

import os
import sys

import argparse
import pickle as pkl
from glob import glob
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
)

# initialize the datasets path
_datapath = get_datapath(_repo_name)


def find_min_max_values(filenames):
    """Finding the minimum and maximum values of the spectrograms
    across all samples in the dataset (PT and DT) to ensure consistent
    scaling across all images.

    Parameters
    ----------
    filenames : list
        List of filenames containing the samples.

    Returns
    -------
    min_val : float
        Minimum value found in the spectrograms.
    max_val : float
        Maximum value found in the spectrograms.
    """

    p_bar = tqdm(total=len(filenames), desc="Finding min and max values")

    min_val = np.inf
    max_val = -np.inf
    for filename in filenames:
        with open(filename, "rb") as f:
            samples = pkl.load(f)
        for sample in samples:
            for su_idx in sample.spectrograms:
                min_val = min(min_val, np.min(sample.spectrograms[su_idx]))
                max_val = max(max_val, np.max(sample.spectrograms[su_idx]))
        del samples  # free memory
        p_bar.update(1)

    p_bar.close()

    print(f"Min value: {min_val}, Max value: {max_val}")
    return min_val, max_val


def flatten_snr_sjr_dicts(labels):
    """Converts the snr_by_su and sjr_by_su nested dicts into separate list columns.

    Transforms nested dicts like {"su_0": [val, val], "su_1": [val, val]} into
    separate keys like "snr_by_su_su_0", "snr_by_su_su_1", etc., enabling
    conversion to CSV format.

    Parameters
    ----------
    labels : dict
        Dictionary containing labels with "snr_by_su" and "sjr_by_su" as nested dicts.

    Returns
    -------
    labels : dict
        Dictionary with flattened SNR/SJR columns and original dicts removed.
    """

    # Flatten snr_by_su
    if "snr_by_su" in labels:
        snr_by_su_dict = labels.pop("snr_by_su")
        for su_key, su_values in snr_by_su_dict.items():
            labels[f"snr_by_su_{su_key}"] = su_values

    # Flatten sjr_by_su
    if "sjr_by_su" in labels:
        sjr_by_su_dict = labels.pop("sjr_by_su")
        for su_key, su_values in sjr_by_su_dict.items():
            labels[f"sjr_by_su_{su_key}"] = su_values

    return labels


def samples_to_imgs_and_labels(
    data_path,
    dataset_nr,
    filename,
    total_sample_idx,
    labels,
    min_val,
    max_val,
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
    min_val : float
        Minimum value found in the spectrograms (for consistent scaling).
    max_val : float
        Maximum value found in the spectrograms (for consistent scaling).
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

        # Collect SNR and SJR data for each SU
        for su_idx in sample.snr_by_su:
            if su_idx not in labels["snr_by_su"]:
                labels["snr_by_su"][su_idx] = []
            labels["snr_by_su"][su_idx].append(sample.snr_by_su[su_idx])

        for su_idx in sample.sjr_by_su:
            if su_idx not in labels["sjr_by_su"]:
                labels["sjr_by_su"][su_idx] = []
            labels["sjr_by_su"][su_idx].append(sample.sjr_by_su[su_idx])

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
            # conversion to the expected range and type for PNG output
            su_spectrogram = (
                (su_spectrogram - min_val) / (max_val - min_val) * 255
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


def parse_arguments():
    """Parse command line arguments.

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
        default=0,
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

    args = parse_arguments()

    dataset_nr = int(args.dataset_number)
    print(f"Processing dataset number: {dataset_nr}")

    total_sample_idx = 0
    labels = {
        "jammer_type": [],
        "jammer_power": [],
        "jammer_location": [],
        "num_legitimate_transmitters": [],
        "snr_by_su": {},
        "sjr_by_su": {},
    }

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

    min_val, max_val = find_min_max_values(filenames)

    # save the minimum and maximum values, so that original values can be restored
    pd.DataFrame({"min_val": [min_val], "max_val": [max_val]}).to_csv(
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
            min_val,
            max_val,
            binary_resource_img=args.binary_resource_img,
        )

    # Flatten nested SNR and SJR dicts into separate columns
    labels = flatten_snr_sjr_dicts(labels)

    pd.DataFrame(labels).to_csv(
        os.path.join(_datapath, f"{dataset_nr}", "labels.csv"),
        index=False,
    )
