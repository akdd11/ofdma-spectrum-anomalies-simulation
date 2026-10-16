"""This script generates the figures of the paper from a generated dataset:

* `snr_sjr`: distribution of the SNR and SJR over all spectrograms
* `allocation`: resource allocation of one sample (the deceptive example)
* `spectrograms`: one example spectrogram per jammer type plus a normal one,
  with the jammed regions highlighted

Only the published data (labels, metadata, and images) is used, so the figures
can be reproduced from the dataset alone. The figures are sized for the IEEE
two-column format and saved as PDF into the `figures` folder of the repository.
The figures shown in the README are additionally saved as PNG into `assets`.

Usage: `python src/plot_figures.py [-d DATASET_NUMBER] [-f FIGURE ...]`
"""

__docformat__ = "numpy"

import os
import sys

import argparse
import re
import hydra
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

# import own modules
_repo_name = "ofdma-spectrum-anomalies-simulation"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]
sys.path.append(os.path.abspath(_module_path))

from src.utils.data_utils import (
    get_datapath,
    get_resource_alloc_img_filenamename,
    get_spectrogram_img_filenamename,
)
from src.utils.jammer_footprint import infer_jammer_boxes

# IEEEtran \columnwidth and \textwidth (identical for A4 and letter) in inches
IEEE_COLUMN_WIDTH = 252 / 72.27
IEEE_TEXT_WIDTH = 516 / 72.27
RATIO = 1.618  # golden ratio

RC_PARAMS = {
    "font.family": "Times New Roman",
    "font.size": 8,
    # embed TrueType instead of Type 3 fonts (required by IEEE PDF eXpress)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
# no creation date, so that regenerating an unchanged figure gives an identical file
PDF_METADATA = {"CreationDate": None}

# Example (sample index, SU index) per jammer type for the spectrogram figure.
# Selected for a moderate local JSNR, such that both the jammer and the
# legitimate signals are visible.
EXAMPLES = {
    "no jammer": (1, 16),
    "barrage": (14001, 15),
    "sweep": (12016, 7),
    "random_hop": (18002, 20),
    "deceptive": (10004, 7),
    "pilot": (16020, 14),
}
PANEL_TITLES = {
    "no jammer": "No jammer",
    "barrage": "Barrage",
    "sweep": "Sweep",
    "random_hop": "Random hopping tone",
    "deceptive": "Deceptive",
    "pilot": "Pilot",
}
# the resource allocation is shown for the sample of the deceptive example, so
# that it can be compared to the legitimate-like jamming signal
ALLOCATION_SAMPLE = EXAMPLES["deceptive"][0]

# Each RE is repeated (nearest neighbour) before embedding the images, so that
# PDF viewers, which smooth images when zooming in, only blur within an RE and
# not across REs. The resolution is chosen such that no RE is lost even if
# matplotlib resamples the image at the export DPI instead of embedding it as is.
UPSAMPLE_SC, UPSAMPLE_SYM = 2, 20
EXPORT_DPI = 1200
README_DPI = 200


class Dataset:
    """Access to the labels, metadata, and images of a generated dataset.

    Parameters
    ----------
    datapath : str
        Path to the dataset folder (containing `labels.csv`, `config.yaml`, ...).
    """

    def __init__(self, datapath):
        self.datapath = datapath
        self.labels = pd.read_csv(os.path.join(datapath, "labels.csv"), low_memory=False)
        min_max = pd.read_csv(os.path.join(datapath, "spectrogram_min_max.csv"))
        self.min_val = min_max["min_val"].values[0]
        self.max_val = min_max["max_val"].values[0]
        with open(os.path.join(datapath, "config.yaml"), "r") as f:
            self.cfg = yaml.safe_load(f)
        self.num_su = len([c for c in self.labels.columns if c.startswith("snr_by_su_")])

    def load_spectrogram(self, sample_idx, su_idx):
        """Spectrogram in dB (rescaled from the 8-bit image), shape (num_sc, num_symbols)"""
        img = Image.open(
            os.path.join(
                self.datapath, "images", get_spectrogram_img_filenamename(sample_idx, su_idx)
            )
        )
        return np.array(img) / 255.0 * (self.max_val - self.min_val) + self.min_val

    def load_allocation(self, sample_idx):
        """Index of the legitimate transmitter per RE (0: unallocated)"""
        img = Image.open(
            os.path.join(
                self.datapath, "images", get_resource_alloc_img_filenamename(sample_idx)
            )
        )
        return np.array(img)

    def per_su_values(self, measure):
        """Finite values of a per SU measure (e.g., 'snr') over all samples and SUs."""
        values = self.labels[[f"{measure}_by_su_{j}" for j in range(self.num_su)]].values.ravel()
        values = values.astype(float)
        # inf for the SJR of samples without jammer
        return values[np.isfinite(values)]


def upsample_for_pdf(img):
    """Repeat each RE, see UPSAMPLE_SC and UPSAMPLE_SYM."""
    return np.repeat(np.repeat(img, UPSAMPLE_SC, axis=0), UPSAMPLE_SYM, axis=1)


def save_figure(fig, filename, num_images=0, min_shape=None, png_filename=None):
    """Save the figure as PDF and check that the images kept their resolution.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Path of the PDF file.
    num_images : int
        Number of image plots (excluding colorbars) to check.
    min_shape : tuple
        Minimum (height, width) the embedded images must have.
    png_filename : str, optional
        If given, the figure is additionally saved as PNG (for the README).
    """
    fig.savefig(filename, dpi=EXPORT_DPI, metadata=PDF_METADATA)
    print(f"Saved {filename}")
    if png_filename is not None:
        fig.savefig(png_filename, dpi=README_DPI, facecolor="white")
        print(f"Saved {png_filename}")
    plt.close(fig)

    if num_images == 0:
        return

    with open(filename, "rb") as f:
        sizes = [(int(h), int(w)) for w, h in re.findall(rb"/Width (\d+)\s*/Height (\d+)", f.read())]
    # the colorbars are smaller than the image plots
    sizes = sorted(sizes, key=lambda hw: hw[0] * hw[1])[-num_images:]
    if min(h for h, _ in sizes) < min_shape[0] or min(w for _, w in sizes) < min_shape[1]:
        print(f"Warning: images were embedded with reduced resolution {sizes}")


def plot_snr_sjr_distribution(dataset, figure_path, assets_path):
    """Empirical distribution of the SNR and SJR over all spectrograms."""
    snr_values = dataset.per_su_values("snr")
    sjr_values = dataset.per_su_values("sjr")
    print(f"Mean SNR: {np.mean(snr_values):.2f} dB, Std SNR: {np.std(snr_values):.2f} dB")
    print(f"Mean SJR: {np.mean(sjr_values):.2f} dB, Std SJR: {np.std(sjr_values):.2f} dB")

    fig, ax = plt.subplots(
        figsize=(IEEE_COLUMN_WIDTH, IEEE_COLUMN_WIDTH / RATIO), layout="constrained"
    )
    sns.histplot(snr_values, label="SNR", linewidth=0, binwidth=0.5, ax=ax)
    sns.histplot(sjr_values, label="SJR", linewidth=0, binwidth=1.0, ax=ax)
    ax.set_xlabel("dB")
    ax.set_xlim(-30, 60)
    ax.legend()
    ax.grid()

    save_figure(fig, os.path.join(figure_path, "snr_sjr_distribution.pdf"))


def allocation_to_rectangles(alloc_rb):
    """Decompose a resource allocation into rectangles of one transmitter each.

    Parameters
    ----------
    alloc_rb : np.ndarray
        Transmitter index per RB and slot (0: unallocated), shape (num_rb, num_slots).

    Returns
    -------
    rectangles : list
        Rectangles (tx, rb0, rb1, slot0, slot1), half-open ranges. Contiguous
        RBs within a slot are merged, and identical RB ranges in consecutive slots.
    """
    rectangles, open_rects = [], {}
    for slot in range(alloc_rb.shape[1] + 1):
        runs = set()
        if slot < alloc_rb.shape[1]:
            col = alloc_rb[:, slot]
            edges = np.flatnonzero(np.diff(col)) + 1
            for rb0, rb1 in zip(np.r_[0, edges], np.r_[edges, len(col)]):
                if col[rb0] > 0:
                    runs.add((int(col[rb0]), int(rb0), int(rb1)))
        # close the rectangles that do not continue in this slot
        for run in list(open_rects):
            if run not in runs:
                rectangles.append((*run, open_rects.pop(run), slot))
        for run in runs:
            open_rects.setdefault(run, slot)
    return rectangles


def plot_resource_allocation(dataset, figure_path, assets_path, sample_idx=ALLOCATION_SAMPLE):
    """Resource allocation of one sample with a discrete colormap per transmitter.

    The allocation is constant per RB and slot, hence it is drawn as vector
    rectangles instead of an image, which PDF viewers would interpolate.
    """
    sc_per_rb = dataset.cfg["subcarriers_per_rb"]
    sym_per_slot = dataset.cfg["symbols_per_slot"]
    n_slots = dataset.cfg["n_slots"]
    num_rb = dataset.cfg["num_rb"]

    alloc = dataset.load_allocation(sample_idx)
    alloc_rb = alloc[::sc_per_rb, ::sym_per_slot]
    assert np.array_equal(
        np.repeat(np.repeat(alloc_rb, sc_per_rb, axis=0), sym_per_slot, axis=1), alloc
    ), "resource allocation is not constant per RB and slot"
    max_tx = int(np.max(alloc_rb))

    # unallocated in white, transmitters in the default color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cmap = mcolors.ListedColormap(["white"] + colors[:max_tx])
    norm = mcolors.BoundaryNorm(np.arange(-0.5, max_tx + 1.5), ncolors=max_tx + 1)

    fig, ax = plt.subplots(
        figsize=(IEEE_COLUMN_WIDTH, IEEE_COLUMN_WIDTH / RATIO), layout="constrained"
    )
    ax.set_facecolor(cmap(norm(0)))
    for tx, rb0, rb1, slot0, slot1 in allocation_to_rectangles(alloc_rb):
        ax.add_patch(
            Rectangle(
                (slot0, rb0), slot1 - slot0, rb1 - rb0, facecolor=cmap(norm(tx)), edgecolor="none"
            )
        )
    ax.set_xlim(0, n_slots)
    ax.set_ylim(0, num_rb)
    ax.set_xlabel("Slot index")
    ax.set_ylabel("Resource block index")
    ax.set_xticks(np.arange(n_slots) + 0.5, labels=range(n_slots))
    ax.set_yticks(range(0, num_rb + 1, 22))

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=range(max_tx + 1))
    cbar.ax.set_yticklabels(["—"] + [str(tx) for tx in range(1, max_tx + 1)])
    cbar.set_label("Transmitter index")

    filename = f"resource_allocation_sample_{sample_idx}"
    save_figure(
        fig,
        os.path.join(figure_path, f"{filename}.pdf"),
        png_filename=os.path.join(assets_path, f"{filename}.png"),
    )


def plot_example_spectrograms(dataset, figure_path, assets_path, examples=EXAMPLES):
    """One example spectrogram per jammer type, jammed regions highlighted."""
    labels = dataset.labels
    sc_per_rb = dataset.cfg["subcarriers_per_rb"]
    sym_per_slot = dataset.cfg["symbols_per_slot"]
    n_slots = dataset.cfg["n_slots"]
    num_rb = dataset.cfg["num_rb"]
    pilot_symbols = dataset.cfg["pilot_slots"]

    num_sc, num_sym = sc_per_rb * num_rb, sym_per_slot * n_slots
    t_per_col = n_slots / num_sym  # in ms (one slot = 1 ms)
    rb_per_row = 1 / sc_per_rb
    pad_t, pad_rb = 0.5 * t_per_col, 2.0  # boxes are padded to keep the jammer visible inside

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(IEEE_TEXT_WIDTH, 3.4),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, letter, (jammer_type, (sample_idx, su_idx)) in zip(
        axes.flat, "abcdef", examples.items()
    ):
        assert labels.loc[sample_idx, "jammer_type"] == jammer_type

        spec = dataset.load_spectrogram(sample_idx, su_idx)
        im = ax.imshow(
            upsample_for_pdf(spec),
            aspect="auto",
            origin="lower",  # row 0 is the lowest frequency
            interpolation="none",
            cmap="viridis",
            vmin=dataset.min_val,
            vmax=dataset.max_val,
            extent=(0, n_slots, 0, num_rb),
        )

        if jammer_type in ("pilot", "deceptive"):
            # grid aligned footprint, identical for all SUs: inferred on the SU
            # that sees the jammer strongest
            jsnr = labels.loc[
                sample_idx, [f"jsnr_local_by_su_{j}" for j in range(dataset.num_su)]
            ].values.astype(float)
            spec_inference = dataset.load_spectrogram(sample_idx, int(np.nanargmax(jsnr)))
        else:
            spec_inference = spec
        boxes = infer_jammer_boxes(
            jammer_type,
            spec_inference,
            dataset.load_allocation(sample_idx),
            sc_per_rb,
            sym_per_slot,
            pilot_symbols,
            occupancy=labels.loc[sample_idx, "jammer_occupancy"],
        )
        for r0, r1, c0, c1 in boxes:
            # pad, but keep the box inside the axes (on the spine if at the edge)
            x0, x1 = max(c0 * t_per_col - pad_t, 0), min(c1 * t_per_col + pad_t, n_slots)
            y0, y1 = max(r0 * rb_per_row - pad_rb, 0), min(r1 * rb_per_row + pad_rb, num_rb)
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=0.8, zorder=3
                )
            )

        ax.set_title(f"({letter}) {PANEL_TITLES[jammer_type]}")
        ax.set_xticks(range(n_slots + 1))
        ax.set_yticks(range(0, num_rb + 1, 22))

        # summary for the caption
        info = (
            f"({letter}) {jammer_type:10s} sample {sample_idx:5d}, SU {su_idx:2d}: "
            f"SNR {labels.loc[sample_idx, f'snr_by_su_{su_idx}']:5.1f} dB"
        )
        if jammer_type != "no jammer":
            info += f", SJR {labels.loc[sample_idx, f'sjr_by_su_{su_idx}']:5.1f} dB"
        print(info)

    for ax in axes[-1]:
        ax.set_xlabel("Time [ms]")
    for ax in axes[:, 0]:
        ax.set_ylabel("RB index")
    fig.colorbar(im, ax=axes, label="PSD [dBm/SC]", aspect=30, pad=0.01)

    save_figure(
        fig,
        os.path.join(figure_path, "example_spectrograms.pdf"),
        num_images=len(examples),
        min_shape=(num_sc, num_sym),
        png_filename=os.path.join(assets_path, "example_spectrograms.png"),
    )


FIGURES = {
    "snr_sjr": plot_snr_sjr_distribution,
    "allocation": plot_resource_allocation,
    "spectrograms": plot_example_spectrograms,
}


def parse_arguments(default_dataset_nr):
    """Parse the command line arguments.

    Parameters
    ----------
    default_dataset_nr : int
        Default dataset number to use if not specified in the command line arguments
        (taken from the dataset_generation.yaml config file).

    Returns
    -------
    args : Namespace
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(description="Generate the figures of the paper.")

    parser.add_argument(
        "-d",
        "--dataset-number",
        default=default_dataset_nr,
        type=int,
        required=False,
        help="Dataset number to plot.",
    )

    parser.add_argument(
        "-f",
        "--figures",
        nargs="+",
        choices=list(FIGURES) + ["all"],
        default=["all"],
        required=False,
        help="Figures to generate (default: all).",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        default=os.path.join(_module_path, "figures"),
        required=False,
        help="Folder the figures are saved to (default: 'figures' of the repository).",
    )

    parser.add_argument(
        "-a",
        "--assets-path",
        default=os.path.join(_module_path, "assets"),
        required=False,
        help="Folder the PNG versions of the figures shown in the README are saved to"
        " (default: 'assets' of the repository).",
    )

    return parser.parse_args()


if __name__ == "__main__":

    # load config file which contains the default dataset number
    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="dataset_generation")

    args = parse_arguments(cfg.dataset_nr)

    datapath = os.path.join(get_datapath(_repo_name), str(args.dataset_number))
    print(f"Plotting dataset: {datapath}")
    dataset = Dataset(datapath)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.assets_path, exist_ok=True)
    figures = list(FIGURES) if "all" in args.figures else args.figures

    with plt.rc_context(RC_PARAMS):
        for name in figures:
            FIGURES[name](dataset, args.output_path, args.assets_path)
