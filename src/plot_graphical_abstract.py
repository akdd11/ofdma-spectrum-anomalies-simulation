"""This script generates the graphical abstract of the paper.

The abstract has three columns, connected by arrows:

* Simulation framework: the rendered scene with the legend of the markers
* Data set: a stack of spectrograms of one sample, as seen by different SUs
* Baseline results: the AUC per jammer type of the supervised and the
  unsupervised baseline as a dot plot

The spectrograms are loaded from the dataset, the AUCs are computed from the
anomaly scores in ``Example_Use/Anomaly_Scores``, and the scene render is taken
from ``figures/scene0_render.png``. The abstract is saved as PDF and PNG into
``figures/graphical_abstract``.

The layout follows the original PowerPoint slide; all positions are given in
inches from the top left corner of the abstract. The slide font is Open Sans;
if it is not installed, it can be passed with ``--font-dir`` (a folder with,
e.g., ``OpenSans-Regular.ttf`` and ``OpenSans-Bold.ttf``), otherwise DejaVu Sans
is used.

Usage: ``python src/plot_graphical_abstract.py [-d DATASET_NUMBER] [--font-dir DIR]``
"""

__docformat__ = "numpy"

import os
import sys

import argparse
import glob
import hydra
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from PIL import Image

# import own modules
_repo_name = "ofdma-spectrum-anomalies-simulation"
_module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.append(os.path.abspath(_module_path))

from src.plot_figures import PDF_METADATA, Dataset, UPSAMPLE_SYM
from src.utils.data_utils import get_dataset_path

# size of the abstract in inches (the content area of the original slide)
WIDTH, HEIGHT = 11.17, 3.9

FONT = "Open Sans"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
# light blue of the slide arrows (PowerPoint text 2 color, 90 % lighter)
ARROW_COLOR = "#dceaf7"
TITLE = "Spectrum Anomaly Detection in OFDMA Systems:\nSimulation Framework and Benchmark Dataset"

# text columns: left edge, heading, and body text (line breaks as on the slide)
COLUMNS = [
    (
        0.21,
        "Simulation framework",
        "including signal generation, ray tracing for\n"
        "realistic propagation in industrial scenario,\n"
        "and five jammer types from naïve to smart",
    ),
    (
        4.63,
        "Data set",
        "of {num_samples} realizations, each with\n"
        "spectrograms for {num_su} sensing unit\n"
        "locations and additional metadata",
    ),
    (
        8.46,
        "Baseline results",
        "for supervised and unsupervised\nlearning",
    ),
]
TITLE_TOP, HEADING_TOP = 0.09, 1.1
LINE_HEIGHT = 1.362  # line height of Open Sans relative to the font size

# scene render, cropped as on the slide (left, top, right, bottom fraction)
RENDER_FILE = os.path.join(_module_path, "figures", "scene0_render.png")
RENDER_CROP = (0.06717, 0.17327, 0.06707, 0.14497)
RENDER_BOX = (0.01, 2.12, 3.07, 1.48)  # x, y, width, height
MARKERS = [  # label, color, x of the marker
    ("Tx", "#00ff58", 0.9),
    ("Jx", "#ff5800", 1.43),
    ("SU", "#fef800", 1.85),
]
MARKER_Y, MARKER_DIAMETER = 3.74, 0.1

# spectrogram stack: one sample, seen by SUs with increasing jammer visibility
# from back to front (local JSNR -14, 6, 34, and 35 dB in dataset 3)
STACK_SAMPLE = 16020  # the pilot example of the paper figure
STACK_SUS = [2, 12, 14, 7]
CARD_BOX = (5.0, 2.21, 1.22, 0.93)  # backmost card: x, y, width, height
CARD_OFFSET = 0.167  # shift of each following card to the bottom right
CARD_BORDER = 1.0  # width of the white card frame in points

ARROWS_X = [3.57, 7.23]
ARROW_Y, ARROW_WIDTH, ARROW_HEIGHT = 2.52, 0.96, 0.64

# AUC dot plot
AUC_BOX = (9.0, 2.06, 2.0, 1.2)  # axes without tick labels: x, y, width, height
SCORE_PATH = os.path.join(_module_path, "Example_Use", "Anomaly_Scores")
# from naive to smart, as in the system model of the paper
JAMMERS = {
    "barrage": "Barrage",
    "sweep": "Sweep",
    "random_hop": "Random hop",
    "deceptive": "Deceptive",
    "pilot": "Pilot",
}
# score file, label file, legend label, and color (as in ROC_Comparison_Jammers.pdf)
METHODS = {
    "supervised_seen": (
        "Scores_Supervised_including_model",
        "Labels_Supervised",
        "Supervised, seen",
        "#1f77b4",
    ),
    "supervised_unseen": (
        "Scores_Supervised_excluding_model",
        "Labels_Supervised",
        "Supervised, unseen",
        "#ff7f0e",
    ),
    "unsupervised": (
        "Scores_Unsupervised_model",
        "Labels_Unsupervised",
        "Unsupervised",
        "#2ca02c",
    ),
}
AUC_LIM = (0.7, 1.005)

PNG_DPI = 600


def setup_font(font_dir=None):
    """Use Open Sans, registered from `font_dir` if given.

    Parameters
    ----------
    font_dir : str, optional
        Folder with the TrueType files of Open Sans.

    Returns
    -------
    family : str
        The font family used.
    """
    if font_dir is not None:
        for font_file in glob.glob(os.path.join(font_dir, "*.ttf")):
            fm.fontManager.addfont(font_file)
    try:
        fm.findfont(fm.FontProperties(family=FONT), fallback_to_default=False)
        return FONT
    except ValueError:
        print(f"Warning: font {FONT} not found, using DejaVu Sans (see --font-dir)")
        return "DejaVu Sans"


def roc_auc(is_anomaly, scores):
    """Area under the ROC curve, computed via the Mann-Whitney U statistic.

    Parameters
    ----------
    is_anomaly : np.ndarray
        Boolean ground truth per sample.
    scores : np.ndarray
        Anomaly score per sample (higher means more anomalous).

    Returns
    -------
    auc : float
        Area under the ROC curve, ties count half.
    """
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    # average rank (starting at 1) of each unique score, i.e., tied scores share a rank
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    num_pos = is_anomaly.sum()
    num_neg = is_anomaly.size - num_pos
    return (ranks[is_anomaly].sum() - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)


def load_auc(jammer, method):
    """AUC of a baseline method on the test set of one jammer type.

    Parameters
    ----------
    jammer : str
        Jammer type, i.e., a key of `JAMMERS`.
    method : str
        Baseline method, i.e., a key of `METHODS`.

    Returns
    -------
    auc : float
        Area under the ROC curve.
    """
    score_file, label_file, _, _ = METHODS[method]
    labels = np.load(os.path.join(SCORE_PATH, jammer, label_file + ".npy")).ravel()
    scores = np.load(os.path.join(SCORE_PATH, jammer, score_file + ".npy")).ravel()
    return roc_auc(labels != "Normal", scores)


def add_axes(fig, x, y, width, height):
    """Add axes at a position given in inches from the top left corner.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The abstract.
    x, y : float
        Top left corner of the axes.
    width, height : float
        Size of the axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The new axes.
    """
    return fig.add_axes((x / WIDTH, 1 - (y + height) / HEIGHT, width / WIDTH, height / HEIGHT))


def draw_text(canvas, num_samples, num_su):
    """Title and the heading and description of each column.

    Parameters
    ----------
    canvas : matplotlib.axes.Axes
        Axes spanning the abstract, in inches from the top left corner.
    num_samples : int
        Number of samples of the dataset.
    num_su : int
        Number of SUs of the dataset.
    """
    canvas.text(
        WIDTH / 2,
        TITLE_TOP,
        TITLE,
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="top",
        linespacing=LINE_HEIGHT,
    )
    for x, heading, body in COLUMNS:
        canvas.text(x, HEADING_TOP, heading, fontsize=16, fontweight="bold", va="top")
        canvas.text(
            x,
            HEADING_TOP + 16 * LINE_HEIGHT / 72,
            body.format(num_samples=num_samples, num_su=num_su),
            fontsize=12,
            va="top",
            linespacing=LINE_HEIGHT,
        )


def draw_arrows(canvas):
    """Arrows between the columns, shaped like the PowerPoint right arrow.

    Parameters
    ----------
    canvas : matplotlib.axes.Axes
        Axes spanning the abstract, in inches from the top left corner.
    """
    head = ARROW_HEIGHT / 2  # length of the arrow head
    for x in ARROWS_X:
        y_mid = ARROW_Y + ARROW_HEIGHT / 2
        shaft_top, shaft_bottom = y_mid - ARROW_HEIGHT / 4, y_mid + ARROW_HEIGHT / 4
        x_head = x + ARROW_WIDTH - head
        canvas.add_patch(
            Polygon(
                [
                    (x, shaft_top),
                    (x_head, shaft_top),
                    (x_head, ARROW_Y),
                    (x + ARROW_WIDTH, y_mid),
                    (x_head, ARROW_Y + ARROW_HEIGHT),
                    (x_head, shaft_bottom),
                    (x, shaft_bottom),
                ],
                facecolor=ARROW_COLOR,
                edgecolor="none",
            )
        )


def draw_scene(fig, canvas):
    """Scene render with the legend of the Tx, Jx, and SU markers.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The abstract.
    canvas : matplotlib.axes.Axes
        Axes spanning the abstract, in inches from the top left corner.
    """
    render = Image.open(RENDER_FILE)
    left, top, right, bottom = RENDER_CROP
    width, height = render.size
    render = render.crop(
        (
            round(left * width),
            round(top * height),
            round((1 - right) * width),
            round((1 - bottom) * height),
        )
    )
    ax = add_axes(fig, *RENDER_BOX)
    ax.imshow(render, aspect="auto")
    ax.axis("off")

    for label, color, x in MARKERS:
        canvas.add_patch(Circle((x, MARKER_Y), MARKER_DIAMETER / 2, color=color, lw=0))
        canvas.text(x + 0.09, MARKER_Y, label, fontsize=10, va="center")


def draw_spectrograms(fig, dataset, sample_idx=STACK_SAMPLE, su_indices=STACK_SUS):
    """Stack of the spectrograms of one sample at several SUs.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The abstract.
    dataset : Dataset
        The dataset to take the spectrograms from.
    sample_idx : int
        Index of the sample. Default: `STACK_SAMPLE`.
    su_indices : list of int
        SUs from the backmost to the front card. Default: `STACK_SUS`.
    """
    x, y, width, height = CARD_BOX
    jammer_type = dataset.labels.loc[sample_idx, "jammer_type"]
    for i, su_idx in enumerate(su_indices):
        ax = add_axes(fig, x + i * CARD_OFFSET, y + i * CARD_OFFSET, width, height)
        spec = dataset.load_spectrogram(sample_idx, su_idx)
        ax.imshow(
            # only 70 symbols: repeat them, such that the time axis is not
            # smoothed across symbols when the image is resampled
            np.repeat(spec, UPSAMPLE_SYM, axis=1),
            aspect="auto",
            origin="lower",  # row 0 is the lowest frequency
            cmap="viridis",
            vmin=dataset.min_val,
            vmax=dataset.max_val,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(CARD_BORDER)
        print(
            f"Card {i}: {jammer_type} sample {sample_idx}, SU {su_idx:2d}, local JSNR "
            f"{dataset.labels.loc[sample_idx, f'jsnr_local_by_su_{su_idx}']:5.1f} dB"
        )


def draw_auc(fig):
    """AUC per jammer type and baseline method as a dot plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The abstract.
    """
    ax = add_axes(fig, *AUC_BOX)
    for i, jammer in enumerate(JAMMERS):
        aucs = {method: load_auc(jammer, method) for method in METHODS}
        assert AUC_LIM[0] <= min(aucs.values()), f"AUC of {jammer} below the axis limit"
        # gray bar spanning the methods, i.e., the gap between them
        ax.plot(
            [min(aucs.values()), max(aucs.values())],
            [i, i],
            color=GRID,
            lw=2.5,
            solid_capstyle="round",
            zorder=1,
        )
        for method, auc in aucs.items():
            _, _, label, color = METHODS[method]
            ax.scatter(
                auc,
                i,
                s=22,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label=label if i == 0 else None,
            )
        print(f"AUC {jammer:10s} " + ", ".join(f"{m} {a:.3f}" for m, a in aucs.items()))

    ax.set_yticks(range(len(JAMMERS)), list(JAMMERS.values()))
    ax.set_ylim(len(JAMMERS) - 0.6, -0.4)  # first jammer on top
    ax.set_xlim(*AUC_LIM)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0], ["0.7", "0.8", "0.9", "1.0"])
    ax.set_xlabel("AUC", labelpad=1)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=MUTED, length=2)
    ax.legend(
        loc="upper left",
        fontsize=7,
        frameon=False,
        handletextpad=0.1,
        labelspacing=0.15,
        borderaxespad=0.1,
        scatterpoints=1,
    )


def plot_graphical_abstract(dataset, output_path):
    """Draw the graphical abstract and save it as PDF and PNG.

    Parameters
    ----------
    dataset : Dataset
        The dataset to take the spectrograms and its size from.
    output_path : str
        Folder the abstract is saved to.
    """
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    # axes spanning the abstract for the text and shapes, in inches from the top left
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, WIDTH)
    canvas.set_ylim(HEIGHT, 0)
    canvas.axis("off")

    draw_text(canvas, num_samples=len(dataset.labels), num_su=dataset.num_su)
    draw_arrows(canvas)
    draw_scene(fig, canvas)
    draw_spectrograms(fig, dataset)
    draw_auc(fig)

    filename = os.path.join(output_path, "graphical_abstract")
    fig.savefig(filename + ".pdf", dpi=PNG_DPI, metadata=PDF_METADATA)
    fig.savefig(filename + ".png", dpi=PNG_DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved {filename}.pdf and .png")


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
    parser = argparse.ArgumentParser(description="Generate the graphical abstract.")

    parser.add_argument(
        "-d",
        "--dataset-number",
        default=default_dataset_nr,
        type=int,
        required=False,
        help="Dataset number to take the spectrograms from.",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        default=os.path.join(_module_path, "figures", "graphical_abstract"),
        required=False,
        help="Folder the abstract is saved to"
        " (default: 'figures/graphical_abstract' of the repository).",
    )

    parser.add_argument(
        "--font-dir",
        default=None,
        required=False,
        help=f"Folder with the TrueType files of {FONT}, if it is not installed.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    # load config file which contains the default dataset number
    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="dataset_generation")

    args = parse_arguments(cfg.dataset_nr)

    datapath = get_dataset_path(_repo_name, args.dataset_number)
    print(f"Using dataset: {datapath}")
    dataset = Dataset(datapath)

    os.makedirs(args.output_path, exist_ok=True)
    rc_params = {
        "font.family": setup_font(args.font_dir),
        "font.size": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        # embed TrueType instead of Type 3 fonts
        "pdf.fonttype": 42,
    }
    with plt.rc_context(rc_params):
        plot_graphical_abstract(dataset, args.output_path)
