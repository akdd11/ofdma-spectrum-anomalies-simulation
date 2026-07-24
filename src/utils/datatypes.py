"""
Contains the datatypes used in the project.
"""

__docformat__ = "numpy"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Transmitter:
    """
    A transmitter is characterized by its location
    and assigned resources.
    """

    def __init__(self, location, resources):
        """
        Initializes a transmitter.

        Parameters
        ----------
        location: tuple
            The location of the transmitter (x,y,z).
        resources: np.ndarray
            Boolean array of resources assigned to the transmitter (RBs).
        """
        self.location = location
        self.resources = resources


class Jammer:
    """
    A jammer is characterized by its location, jammed resources and
    the transmit power of the jammer.
    """

    def __init__(
        self,
        location,
        orientation,
        transmit_power,
        type,
        antenna_pattern,
    ):
        """
        Initializes a jammer.

        Parameters
        ----------
        location: tuple
            The location of the jammer (x,y,z).
        orientation: tuple
            The orientation of the jammer, corresponding to roatation around
            z, y, and x axis in radians.
        transmit_power: float
            The transmit power of the jammer.
        type: str
            The jammer type (e.g., barrage, sweep, deceptive).
        antenna_pattern: str
            Name of the antenna pattern used by the jammer.
        """
        self.location = location
        self.orientation = orientation
        self.transmit_power = transmit_power
        self.type = type
        self.antenna_pattern = antenna_pattern


class Sample:
    """
    A sample is characterized by the transmitters, potentially
    the jammer and the assigned resources.

    Particularly it contains spectrograms for each SU.

    Parameters
    ----------
    transmitters: list
        List of Transmitter objects.
    jammers: list
        List of Jammer objects.
    spectrograms: dict
        Dictionary containing the spectrogram of each SU.
        The spectrogram only contains the magnitude in logarithmic domain,
        i .e., it is real-valued and the unit is dBm.
    sjr_by_su: dict
        Signal-to-jammer ratio per SU in dB, averaged over the whole
        time-frequency grid.
    snr_by_su: dict
        Signal-to-noise ratio per SU in dB, averaged over the whole
        time-frequency grid.
    jammer_occupancy: float
        Fraction of the resource elements of the time-frequency grid that are
        occupied by the jammer. The jammer footprint does not depend on the SU,
        hence this is a sample level quantity. NaN if there is no jammer.
    jsnr_local_by_su: dict
        Jammer-to-signal-plus-noise ratio per SU in dB, evaluated only on the
        resource elements occupied by the jammer. Positive values indicate that
        the jammer dominates its own footprint.
    db_contrast_global_by_su: dict
        Mean change of the spectrogram in dB caused by the jammer, averaged
        over all resource elements. This corresponds to a mean pooled anomaly
        score, i.e., a sparse jammer is diluted in the same way.
    db_contrast_local_by_su: dict
        Mean change of the spectrogram in dB caused by the jammer, averaged
        only over the resource elements occupied by the jammer.
    """

    def __init__(self):
        """
        Initializes a sample.
        """

        self.transmitters = []
        self.jammers = []
        self.spectrograms = {}
        self.noise_power_per_su = {}
        self.sjr_by_su = {}
        self.snr_by_su = {}
        self.jammer_occupancy = np.nan
        self.jsnr_local_by_su = {}
        self.db_contrast_global_by_su = {}
        self.db_contrast_local_by_su = {}

    def add_transmitter(self, transmitter):
        """
        Adds a transmitter to the sample.

        Parameters
        ----------
        transmitter: Transmitter
            The transmitter to be added.
        """

        self.transmitters.append(transmitter)

    def add_jammer(self, jammer):
        """
        Adds a jammer to the sample.

        Parameters
        ----------
        jammer: Jammer
            The jammer to be added.
        """

        self.jammers.append(jammer)

    def add_spectrogram(self, su_idx, spectrogram):
        """
        Adds a spectrogram for a SU.

        Parameters
        ----------
        su_idx: int
            The index of the SU.
        spectrogram: np.ndarray
            The spectrogram of the SU.
            Saved as float32 to reduce memory usage.
        """

        self.spectrograms[su_idx] = spectrogram.astype(np.float32)

    def plot_spectrogram(self, su_idx, cbar=True, vmin=None, vmax=None, ax=None):
        """
        Plots the spectrogram of a SU.

        Parameters
        ----------
        su_idx: int
            The name of the SU.
        cbar: bool
            Whether to show the colorbar or not.
        vmin: float
            The minimum value for the colorbar.
        vmax: float
            The maximum value for the colorbar.
        ax: matplotlib.axes.Axes
            The axes to plot the spectrogram on. Only if several spectrograms shall
            be plotted in the same figure.
        """

        if su_idx not in self.spectrograms:
            raise ValueError(f"No spectrogram available for SU {su_idx}.")

        if vmin is None:
            vmin = np.min(self.spectrograms[su_idx])
        if vmax is None:
            vmax = np.max(self.spectrograms[su_idx])

        sns.heatmap(
            self.spectrograms[su_idx],
            cbar=cbar,
            cbar_kws={"label": "Power [dBm]"},
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

        if ax is None:
            plt.title(f"SU {su_idx}")
            plt.xlabel("Symbol")
            plt.ylabel("Subcarrier")
            plt.show()

    def plot_all_spectrograms(self):
        """
        Plots the spectrogram for all SUs of the sample.
        """

        if len(self.spectrograms) not in [12, 24]:
            raise ValueError("Only implemeneted for 12 SUs.")

        fig, ax = plt.subplots(2, 6, figsize=(18, 8))

        # find minimum and maximum values for consistent plotting
        vmin = np.min(
            [np.min(self.spectrograms[su_idx]) for su_idx in self.spectrograms]
        )
        vmax = np.max(
            [np.max(self.spectrograms[su_idx]) for su_idx in self.spectrograms]
        )

        for su_idx in self.spectrograms:
            ax_idx = ((su_idx // 6) % 2, su_idx % 6)
            if su_idx % 6 == 5:
                plot_cbar = True
            else:
                plot_cbar = False
            self.plot_spectrogram(
                su_idx, cbar=plot_cbar, vmin=vmin, vmax=vmax, ax=ax[ax_idx]
            )
            ax[ax_idx].set_title(f"SU {su_idx}")

            ax[ax_idx].invert_yaxis()

            if ax_idx[0] == 1:
                ax[ax_idx].set_xlabel("Symbol")
            else:
                ax[ax_idx].set_xticklabels([])

            if ax_idx[1] == 0:
                ax[ax_idx].set_ylabel("Subcarrier")
            else:
                ax[ax_idx].set_yticklabels([])

            if su_idx % 12 == 11:
                plt.tight_layout()
                plt.show()

                if su_idx != len(self.spectrograms) - 1:
                    # start a new plot if there 24 SUs and the first half is finished
                    fig, ax = plt.subplots(2, 6, figsize=(18, 8))
