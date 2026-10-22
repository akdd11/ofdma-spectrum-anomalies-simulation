"""
This file contains the utilities to handle ray tracing, scenes, etc.
"""

__docformat__ = "numpy"

import os

import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import scipy.signal as scsig
import yaml

import sionna.rt as srt

from utils.ofdm_utils import (
    generate_user_signal_freq,
    generate_jammer_signal_freq,
    freq_signal_to_time_signal,
    calc_complex_spectrogram,
    calc_spectrogram,
    crop_spectrogram_to_bandwidth,
    jammer_mask_on_stft_grid,
)
from utils import impairment_utils


def calc_noise_power_dbm(bandwidth_hz, noise_figure_db=0, additional_impairments_db=0):
    """Calculate the noise power in dBm for a given bandwidth and noise figure.

    Parameters
    ----------
    bandwidth_hz : float
        Bandwidth in Hz.
    noise_figure_db : float, optional
        Noise figure in dB. Default: 0.
    additional_impairments_db : float, optional
        Additional impairments in dB to prevent overly optimistic SNR values. Default: 0.


    Returns
    -------
    noise_power_dbm : float
        Noise power in dBm.
    """
    k = 1.380649e-23  # Boltzmann constant in J/K
    T = 290  # Standard temperature in K

    noise_power_dbm = (
        10 * np.log10(k * T * bandwidth_hz)
        + 30
        + noise_figure_db
        + additional_impairments_db
    )

    return noise_power_dbm


def init_scene(scene_path, f_c, su_coordinates):
    """Initialize the Sionna scene.

    This includes the following steps:
    * Load the scene from file
    * Configure SU antennas
    * Add the sensing units

    Parameters
    ----------
    scene_path : str
        Path to the scene file.
    f_c : float
        Carrier frequency.
    su_coordinates : list
        List of sensing unit coordinates.

    Returns
    -------
    scene : sionna.rt.Scene
        Scene object.
    """
    scene = srt.load_scene(scene_path, merge_shapes=False)
    scene.frequency = f_c

    COLOR_CONCRETE = (0.6, 0.6, 0.6)
    COLOR_METAL = (0.7, 0.7, 0.75)

    # define the materials
    material_metal = srt.ITURadioMaterial(
        name="metal", itu_type="metal", thickness=0.02, color=COLOR_METAL
    )
    material_concrete = srt.ITURadioMaterial(
        name="concrete-room", itu_type="concrete", thickness=0.2, color=COLOR_CONCRETE
    )
    material_concrete_ceiling = srt.ITURadioMaterial(
        name="concrete-ceiling",
        itu_type="concrete",
        thickness=0.2,
        color=COLOR_CONCRETE,
    )

    # assign the radio materials
    for obj_name in scene.objects:
        if "room" in obj_name:
            if "ceiling" in obj_name:
                scene.objects[obj_name].radio_material = material_concrete_ceiling
            else:
                scene.objects[obj_name].radio_material = material_concrete
        else:
            scene.objects[obj_name].radio_material = material_metal

    # Antennas for sensing units
    scene.rx_array = srt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    # Add the sensing units
    for idx_su, su_coord in enumerate(su_coordinates):
        su = srt.Receiver(name=f"su{idx_su}", position=su_coord)
        scene.add(su)

    return scene


def get_scene_size(scene):
    """Get the size of the scene.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Scene object.

    Returns
    -------
    scene_size : tuple
        Size of the scene.
    """

    return (scene.mi_scene.bbox().max - scene.mi_scene.bbox().min).numpy()


def configure_tx_antenna_pattern(scene, tx_antenna_pattern):
    """Configure the antenna pattern for the currently considered transmitter.

    This applies to both legitimate transmitters and jammers.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Scene object.
    tx_antenna_pattern : str
        Antenna pattern for the regular transmitters or jammers.
        Needs to be supported by Sionna (e.g., "iso" or "tr38901").

    Returns
    -------
    scene : sionna.rt.Scene
        Updated scene object with configured antenna pattern.
    """

    # Antenna for regular transmitters and jammers
    scene.tx_array = srt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=tx_antenna_pattern,
        polarization="V",
    )

    return scene


def get_random_tx_location(scene_nr, scene_size, tx_height=1.5, **kwargs):
    """Return a random location for a transmitter not enclosed within any obstacle.

    Parameters
    ----------
    scene_nr : int
        Index of the current scene.
    scene_size : tuple
        Size of the scene.
    tx_height : float, optional
        Height of the transmitter from the ground.
        Default: 1.5.

    Keyword Arguments
    -----------------
    su_coordinates : np.array, optional
        List of sensing unit coordinates (list of 3D coordinates).
        To avoid unrealistic ray tracing results by placing the
        transmitter too close to the SU, a minimum distance between
        the transmitter and the SU is ensured to be kept if specified.
    min_dist_to_su : float, optional
        Minimum distance between the transmitter and the closest
        sensing unit in meters. Required if ``su_coordinates`` is given.

    Returns
    -------
    tx_pos : np.ndarray
        Array containing x, y and z coordinates of the transmitter.
    """

    if "su_coordinates" in kwargs:
        if "min_dist_to_su" in kwargs:
            min_dist_to_su = kwargs["min_dist_to_su"]
        else:
            raise ValueError("Minimum distance to SU not specified.")
        su_coordinates = kwargs["su_coordinates"]
    elif "min_dist_to_su" in kwargs:
        raise ValueError("Minimum distance to SU specified without SU coordinates.")

    while True:
        tx_pos = [
            np.random.uniform(0, scene_size[0]),
            np.random.uniform(0, scene_size[1]),
            tx_height,
        ]
        if enclosed_in_obstacle(tx_pos, scene_nr):
            continue
        elif "min_dist_to_su" in kwargs:
            min_dist = np.min(np.linalg.norm(su_coordinates - tx_pos, axis=1))
            if min_dist < min_dist_to_su:
                continue

        break

    return np.array(tx_pos)


def enclosed_in_obstacle(tx_pos, scene_nr):
    """Check if the transmitter is enclosed within any obstacle.

    Parameters
    ----------
    tx_pos : list
        List containing x, y and z coordinates of the transmitter.
    scene_nr : int
        Index of the current scene.

    Returns
    -------
    enclosed : bool
        True if the transmitter is enclosed within an obstacle.
    """

    # Only read the scene description if it has not been read before
    if not hasattr(enclosed_in_obstacle, "_obstacles"):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "scenes",
            f"scene{scene_nr}",
            "scene_attributes.yaml",
        )

        # Load the YAML file
        with open(path, "r") as file:
            scene_description = yaml.safe_load(file)
        obstacles = scene_description.get("obstacles", {})

        # Attach the obstacles to the function
        enclosed_in_obstacle._obstacles = obstacles

    if enclosed_in_obstacle._obstacles == {}:  # No obstacles
        return False

    for obstacle in enclosed_in_obstacle._obstacles.values():
        x, y, z = obstacle["anchor_point"]
        w, l, h = obstacle["edge_length"]
        if x <= tx_pos[0] <= (x + w) and y <= tx_pos[1] <= (y + l) and tx_pos[2] <= h:
            return True
        else:
            continue

    return False


def plot_scenario2D(scene_nr, sample, add_device_labels=False, su_coordinates=[]):
    """Plot the scenario (radio devices, obstacles) in 2D.

    Parameters
    ----------
    scene_nr : int
        Index of the current scene to load the obstacle description and scene size.
    sample : Sample
        Sample object containing the transmitters and jammers.
    add_device_labels : bool, optional
        If True, the device labels are added to the plot.
        Default: False.
    su_coordinates : list, optional
        List of sensing unit coordinates.
        Default: [].

    """

    fig, ax = plt.subplots()

    # Only read the scene description if it has not been read before
    if not hasattr(plot_scenario2D, "_scene_description"):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "scenes",
            f"scene{scene_nr}",
            "scene_attributes.yaml",
        )

        # Load the YAML file
        with open(path, "r") as file:
            scene_description = yaml.safe_load(file)

        # Attach the obstacles to the function
        plot_scenario2D._scene_description = scene_description

    obstacles = plot_scenario2D._scene_description.get("obstacles", {})

    for obstacle in obstacles.values():
        x, y = obstacle["anchor_point"][:2]
        w, l = obstacle["edge_length"][:2]

        rect = patches.Rectangle((x, y), w, l, linewidth=1, edgecolor="black")
        ax.add_patch(rect)

    for tx_idx, tx in enumerate(sample.transmitters):
        x, y = tx.location[:2]
        plt.plot(x, y, "green", marker="x", markersize=5, linewidth=10)
        if add_device_labels:
            plt.text(x, y, f"TX{tx_idx}", fontsize=8)

    for jam_idx, jammer in enumerate(sample.jammers):
        x, y = jammer.location[:2]
        plt.plot(x, y, "red", marker="x", markersize=5, linewidth=10)
        if add_device_labels:
            plt.text(x, y, f"JAM{jam_idx}", fontsize=8)

    for su_idx, su in enumerate(su_coordinates):
        x, y = su[:2]
        plt.plot(x, y, "yellow", marker="o", markersize=5, linewidth=10)
        if add_device_labels:
            if su_idx < 12:
                plt.text(x, y, f"SU{su_idx}", fontsize=8)
            else:
                plt.text(x, y + 0.8, f"SU{su_idx}", fontsize=8)

    room = plot_scenario2D._scene_description.get("room", {})
    w, l = room["size"][:2]
    plt.xlim(0, w)
    plt.ylim(0, l)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    ax.set_aspect("equal")

    plt.title(
        f"No. regular transmitters: {len(sample.transmitters)}, "
        f"No. jammers: {len(sample.jammers)}"
    )
    plt.show()


def get_su_coordinates(module_path):
    """Generate coordinates of the sensing units.

    The SUs are placed on the left and right side of every of the six
    (or in a vertical line above).

    Parameters
    ----------
    module_path : str
        Path to the module, used to load the scene description for checking
        if the sensing units are placed within an obstacle.
    Returns
    -------
    su_coords : list
        List of measurement points for the sensing units.
    """

    su_coordinates_filename = os.path.join(
        module_path, "src", "conf", "su_coordinates.yaml"
    )

    with open(su_coordinates_filename, "r") as file:
        su_coordinates_data = yaml.safe_load(file)

    su_coords = su_coordinates_data.get("su_coordinates", [])

    return su_coords


def plot_frequency_responses(fft_freq, h, num_jammers=0):
    """Plot the frequency responses for each transmitter / SU combination.

    Parameters
    ----------
    fft_freq : np.ndarray
        Array of the FFT frequencies.
    h : np.ndarray
        Array of the channel impulse responses.
    num_jammers : int, optional
        Number of jammers (required for labeling).
        Default: 0.
    """

    num_su = h.shape[0]
    num_tx_total = h.shape[2]

    if num_su not in [12, 24]:
        raise ValueError("Only 12 or 24 SUs are supported for visualization.")

    fig, ax = plt.subplots(2, 6, figsize=(18, 8))

    # find minimum and maximum values to ensure consistent plotting over all SUs
    vmin = np.inf
    vmax = -np.inf
    for su_idx in range(h.shape[0]):
        for tx_idx in range(h.shape[2]):
            # some workaround to ignore the -inf in axis limits
            tmp_arr = 20 * np.log10(np.abs(h[su_idx, 0, tx_idx, 0, 0, :]))
            tmp_arr[tmp_arr == -np.inf] = np.nan
            vmin = np.nanmin([vmin, np.min(tmp_arr)])
            vmax = np.nanmax(
                [vmax, np.max(20 * np.log10(np.abs(h[su_idx, 0, tx_idx, 0, 0, :])))]
            )
    vmin = vmin - 1  # add some margin
    vmax = vmax + 1

    for su_idx in range(num_su):
        ax_idx = ((su_idx // 6) % 2, su_idx % 6)

        for tx_idx in range(num_tx_total):
            if tx_idx < num_tx_total - num_jammers:
                label = f"TX{tx_idx}"
            else:
                label = f"JAM{tx_idx-num_tx_total+num_jammers}"
            if np.sum(np.abs(h[su_idx, 0, tx_idx, 0, 0, :])) == 0:
                continue
            ax[ax_idx].plot(
                fft_freq.numpy(),
                20 * np.log10(np.abs(h[su_idx, 0, tx_idx, 0, 0, :])),
                label=label,
                color=f"C{tx_idx}",
            )
        ax[ax_idx].set_title(f"SU{su_idx}")
        ax[ax_idx].legend()
        ax[ax_idx].set_ylim([vmin, vmax])

        if ax_idx[0] == 1:
            ax[ax_idx].set_xlabel("Frequency [Hz]")
        else:
            ax[ax_idx].set_xticklabels([])

        if ax_idx[1] == 0:
            ax[ax_idx].set_ylabel("Frequency response [dB]")
        else:
            ax[ax_idx].set_yticklabels([])

        if su_idx % 12 == 11:
            plt.tight_layout()
            plt.show()

            if su_idx != num_su - 1:
                # start a new plot if there 24 SUs and the first half is finished
                fig, ax = plt.subplots(2, 6, figsize=(18, 8))


def scale_time_signal_to_target_power(time_signal, target_power):
    """Scale the time signal to the target power.

    Parameters
    ----------
    time_signal : np.ndarray
        Time signal to be scaled.
    target_power : float
        Target power in dBm.

    Returns
    -------
    time_signal : np.ndarray
        Scaled time signal with target power.
    """
    # normalizing the signal to unit power (denominator) and multiply with desired amplitude (numerator)
    scaling_factor = np.sqrt(
        10 ** (target_power / 10)
        / (np.mean(np.abs(time_signal[time_signal != 0]) ** 2))
    )
    logging.debug(f"Scaling factor: {scaling_factor:.2f}")
    time_signal = time_signal * scaling_factor

    return time_signal


def channel_tap_range(paths_list, sampling_frequency, margin):
    """Range of time lags covering the delays of all given path sets.

    The range is shared by all path sets of a sample, so that the resulting
    channel taps can be stacked along the transmitter axis.

    Parameters
    ----------
    paths_list : list of sionna.rt.Paths
        Path sets computed by the path solver.
    sampling_frequency : float
        Sampling frequency of the simulated time signals in Hz.
    margin : int
        Number of sinc taps kept before lag 0 and after the largest path delay.

    Returns
    -------
    l_min : int
        Smallest time lag in samples (negative).
    l_max : int
        Largest time lag in samples.
    """
    max_delay = 0.0
    for paths in paths_list:
        tau = paths.tau.numpy()
        if tau.size > 0:
            max_delay = max(max_delay, float(np.max(tau)))  # invalid paths have tau = -1

    return -margin, int(np.ceil(max_delay * sampling_frequency)) + margin


def compute_channel_taps(paths, sampling_frequency, l_min, l_max):
    """Discrete baseband channel taps on the sample grid of the simulation.

    Computed with sinc interpolation of the absolute path delays (Sionna's
    `Paths.taps`). In contrast to the IFFT of the CFR, the taps are not
    circular: the pre-cursor sidelobes of fractional delays sit at negative
    lags and do not wrap around to lags of about `nfft`, where they would place
    a copy of each OFDM symbol into the window of the next one.

    Parameters
    ----------
    paths : sionna.rt.Paths
        Paths computed by the path solver.
    sampling_frequency : float
        Sampling frequency of the simulated time signals in Hz. It is also the
        bandwidth the channel is limited to, so that the taps are spaced by one
        sample.
    l_min : int
        Smallest time lag in samples, must not be positive.
    l_max : int
        Largest time lag in samples.

    Returns
    -------
    taps : np.ndarray
        Complex channel taps for the lags `l_min, ..., l_max`. Shape according
        to sionna output, i.e.,
        [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1].
    """
    return paths.taps(
        bandwidth=sampling_frequency,
        l_min=l_min,
        l_max=l_max,
        normalize=False,
        normalize_delays=False,
        out_type="numpy",
    )


def apply_channel_taps(signal, taps, l_min, out_len):
    """Convolve a time signal with channel taps starting at lag `l_min`.

    Parameters
    ----------
    signal : np.ndarray
        Transmitted time signal.
    taps : np.ndarray
        Channel taps for the lags `l_min, l_min + 1, ...`.
    l_min : int
        Time lag of the first tap in samples, must not be positive.
    out_len : int
        Length of the returned signal.

    Returns
    -------
    received : np.ndarray
        Received time signal, aligned with the transmitted one (lag 0 maps to
        the same sample index) and truncated to `out_len` samples.
    """
    assert l_min <= 0, "the first tap must not lie after lag 0"
    return scsig.convolve(signal, taps)[-l_min : -l_min + out_len]


def taps_to_cfr(taps, l_min, nfft):
    """Channel frequency response of channel taps on the FFT grid.

    Parameters
    ----------
    taps : np.ndarray
        Channel taps along the last axis for the lags `l_min, l_min + 1, ...`.
    l_min : int
        Time lag of the first tap in samples.
    nfft : int
        Number of FFT points.

    Returns
    -------
    h : np.ndarray
        Channel frequency response with zero frequency in the centre, same
        shape as `taps` except for the last axis, which has `nfft` entries.
    """
    lags = np.arange(l_min, l_min + taps.shape[-1]) % nfft
    cir = np.zeros(taps.shape[:-1] + (nfft,), dtype=complex)
    for tap_idx, lag in enumerate(lags):
        cir[..., lag] += taps[..., tap_idx]

    return np.fft.fftshift(np.fft.fft(cir, axis=-1), axes=-1)


def create_spectrograms(sample, cfg, h, l_min, su_hardware, noise=False):
    """Create spectrograms for a sample with the given channel taps for each SU.

    Parameters
    ----------
    sample : Sample
        Sample object containing the transmitters and jammers.
    cfg : OmegaConf
        Configuration object.
    h : np.ndarray
        Array of the complex channel taps, see `compute_channel_taps`.
        Shape [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_taps],
        the jammer being the last transmitter.
    l_min : int
        Time lag of the first tap in samples, must not be positive.
    su_hardware : pd.DataFrame
        Per-SU hardware impairment parameters, as returned by
        `impairment_utils.generate_su_hardware_params`. Only read from if
        the corresponding impairment is enabled in ``cfg.impairments``.
    noise : bool, optional
        If True, the spectrogram is initialized with noise. Otherwise,
        the spectrogram is initialized with zeros (linear domain) and
        the mean noise level is considered at the end.
        Default: False.

    Returns
    -------
    sample : Sample
        Updated sample object with spectrograms added. In addition, the SNR and
        SJR per SU, as well as the jammer occupancy, the local JSNR and the dB
        contrast are stored in the sample. See the Sample class for details on
        those metrics.
    """

    # hardware impairments, gated on the master
    # switch and their individual enabled flags
    impairments_on = bool(cfg.impairments.enabled)
    timing_offset_enabled = impairments_on and cfg.impairments.su_timing_offset.enabled
    lo_offset_enabled = impairments_on and cfg.impairments.su_lo_freq_offset.enabled
    dc_offset_enabled = impairments_on and cfg.impairments.rx_dc_offset.enabled
    iq_imbalance_enabled = impairments_on and cfg.impairments.rx_iq_imbalance.enabled

    sample_rate_hz = cfg.subcarrier_spacing * cfg.nfft

    # create the plain user signals, first in time domain
    # and then convert to frequency domain
    user_signals_time = {}
    for tx_idx in range(len(sample.transmitters)):

        logging.debug(f"Generating user signal for TX{tx_idx}.")

        user_signal_freq = generate_user_signal_freq(
            sample.transmitters[tx_idx].resources,
            cfg.subcarriers_per_rb,
            cfg.symbols_per_slot,
            np.random.choice(cfg.bits_per_symbol),
        )
        # convert the user signal to time domain
        user_signal_time = freq_signal_to_time_signal(
            user_signal_freq,
            cfg.nfft,
            cfg.cp_len,
            cfg.idx_first_sc,
            add_cp=True,
        )

        # scale for the desired TX power
        user_signals_time[tx_idx] = scale_time_signal_to_target_power(
            user_signal_time, cfg.tx_power
        )

    # if required, generate the jammer signal
    if len(sample.jammers) == 1:
        if sample.jammers[0].type == "pilot":
            allocated_resources = {}
            for tx_idx in range(len(sample.transmitters)):
                allocated_resources[tx_idx] = sample.transmitters[tx_idx].resources
            jammer_signal_freq, add_cp_for_jammer = generate_jammer_signal_freq(
                sample.jammers[0].type, cfg, allocated_resources=allocated_resources
            )
        elif sample.jammers[0].type == "deceptive":
            jammer_signal_freq, add_cp_for_jammer = generate_jammer_signal_freq(
                sample.jammers[0].type, cfg, num_tx=len(sample.transmitters)
            )
        else:
            jammer_signal_freq, add_cp_for_jammer = generate_jammer_signal_freq(
                sample.jammers[0].type, cfg
            )

        if cfg.plots.jammer_spectrogram:
            calc_spectrogram(
                freq_signal_to_time_signal(
                    jammer_signal_freq,
                    cfg.nfft,
                    cfg.cp_len,
                    cfg.idx_first_sc,
                    add_cp=add_cp_for_jammer,
                ),
                cfg.nfft,
                cfg.nfft + cfg.cp_len,
                dynamic_range=50,  # dynamic range in dB for plotting
                plot=True,
            )

        jammer_signal_time = freq_signal_to_time_signal(
            jammer_signal_freq,
            cfg.nfft,
            cfg.cp_len,
            cfg.idx_first_sc,
            add_cp=add_cp_for_jammer,
        )

        if not add_cp_for_jammer:
            # signal needs to be cutted to make sure the length fits exactly
            signal_len = (cfg.nfft + cfg.cp_len) * cfg.n_slots * cfg.symbols_per_slot
            jammer_signal_time = jammer_signal_time[:signal_len]

        jammer_signal_time = scale_time_signal_to_target_power(
            jammer_signal_time, sample.jammers[0].transmit_power
        )

    elif len(sample.jammers) > 1:
        raise ValueError("Only one jammer is currently supported.")

    signal_len = (cfg.nfft + cfg.cp_len) * cfg.n_slots * cfg.symbols_per_slot

    # iterate over the SUs to create the spectrograms
    for su_idx in range(h.shape[0]):

        logging.debug(f"Creating spectrogram for SU{su_idx}.")

        # per-SU hardware impairment parameters/derived quantities for this
        # sample. Fixed per SU, only computed/applied if enabled.
        # Residual sub-CP timing error (5G-UE-grade sync). It is bounded
        # well below the CP, so the full column count still fits and the image
        # dimensions match the grid-synchronous baseline (num_cols=None -> the
        # default 70 columns, no column dropped).
        su_num_cols = None
        if timing_offset_enabled:
            su_timing_offset = int(su_hardware.loc[su_idx, "timing_offset_samples"])
        else:
            su_timing_offset = 0

        if lo_offset_enabled:
            su_delta_f_hz = (
                su_hardware.loc[su_idx, "lo_freq_offset_ppm"] * 1e-6 * cfg.f_c
            )
            su_lo_rotation = impairment_utils.lo_frequency_rotation(
                su_delta_f_hz, sample_rate_hz, signal_len
            )

        if iq_imbalance_enabled:
            su_iq_alpha, su_iq_beta = impairment_utils.compute_iq_imbalance_coeffs(
                su_hardware.loc[su_idx, "iq_gain_imbalance_db"],
                su_hardware.loc[su_idx, "iq_phase_imbalance_deg"],
            )

        # --- legitimate signal -----------------------------------------------
        # The SU antenna receives the sum of all transmitter signals as a
        # single time-domain waveform; the spectrogram is computed once on that
        # sum. The STFT is linear, so this is identical to the previous
        # per-emitter spectrogram summation, while any per-emitter time-domain
        # impairment still slots in before the sum.
        signal_time = np.zeros(signal_len, dtype=complex)
        for tx_idx in range(len(sample.transmitters)):

            signal_time += apply_channel_taps(
                user_signals_time[tx_idx], h[su_idx, 0, tx_idx, 0, 0, :], l_min, signal_len
            )

        if lo_offset_enabled:
            signal_time = signal_time * su_lo_rotation

        if iq_imbalance_enabled:
            signal_time = impairment_utils.apply_iq_imbalance(
                signal_time, su_iq_alpha, su_iq_beta
            )

        signal_spec = calc_complex_spectrogram(
            signal_time,
            cfg.nfft,
            cfg.nfft + cfg.cp_len,
            offset=su_timing_offset,
            num_cols=su_num_cols,
        )
        signal_spec = crop_spectrogram_to_bandwidth(
            signal_spec, cfg.idx_first_sc, cfg.num_subcarriers
        )

        signal_power = np.mean(np.sum(np.abs(signal_spec) ** 2, axis=0))

        # total received spectrogram; jammer and noise are added on top.
        # signal_spec is kept as the jammer- and noise-free reference required
        # to quantify the impact of the jammer on the spectrogram further below.
        total_spec = signal_spec.copy()

        # --- jammer ----------------------------------------------------------
        if len(sample.jammers) == 1:

            if su_idx == 0:
                # the resource elements occupied by the jammer are identical for
                # all SUs, hence the occupancy is a sample level quantity. It is
                # mapped onto the network-synchronous STFT grid by time (the
                # per-SU sub-CP timing residual does not move a window centre
                # across a block boundary by more than a few samples)
                jammer_mask = jammer_mask_on_stft_grid(
                    jammer_signal_freq,
                    add_cp_for_jammer,
                    cfg.nfft,
                    cfg.cp_len,
                    signal_spec.shape[1],
                )
                sample.jammer_occupancy = float(np.mean(jammer_mask))

            # assuming the channel of the jammer is the last one in the axis of the transmitters
            jammer_signal_time_after_channel = apply_channel_taps(
                jammer_signal_time, h[su_idx, 0, -1, 0, 0, :], l_min, signal_len
            )

            if lo_offset_enabled:
                jammer_signal_time_after_channel = (
                    jammer_signal_time_after_channel * su_lo_rotation
                )

            if iq_imbalance_enabled:
                jammer_signal_time_after_channel = impairment_utils.apply_iq_imbalance(
                    jammer_signal_time_after_channel, su_iq_alpha, su_iq_beta
                )

            jammer_spec = calc_complex_spectrogram(
                jammer_signal_time_after_channel,
                cfg.nfft,
                cfg.nfft + cfg.cp_len,
                offset=su_timing_offset,
                num_cols=su_num_cols,
            )
            jammer_spec = crop_spectrogram_to_bandwidth(
                jammer_spec, cfg.idx_first_sc, cfg.num_subcarriers
            )

            jammer_power = np.mean(np.sum(np.abs(jammer_spec) ** 2, axis=0))
            sample.sjr_by_su[su_idx] = 10 * np.log10(signal_power / jammer_power)

            total_spec += jammer_spec
        else:
            sample.sjr_by_su[su_idx] = np.inf

        # add noise according to a specified SNR. The noise level relates to the weakest signal
        if noise:
            noise_signal = np.random.normal(
                size=signal_len
            ) + 1j * np.random.normal(size=signal_len)
            noise_signal = scale_time_signal_to_target_power(
                noise_signal, cfg.p_noise_dbm
            )

            if dc_offset_enabled:
                # folded into the noise/background term rather than any
                # signal or jammer component
                noise_signal = impairment_utils.apply_dc_offset(
                    noise_signal,
                    cfg.nfft,
                    su_hardware.loc[su_idx, "dc_offset_over_noise_db"],
                    su_hardware.loc[su_idx, "dc_offset_phase_rad"],
                )

            noise_spec = calc_complex_spectrogram(
                noise_signal,
                cfg.nfft,
                cfg.nfft + cfg.cp_len,
                offset=su_timing_offset,
                num_cols=su_num_cols,
            )
            noise_spec = crop_spectrogram_to_bandwidth(
                noise_spec, cfg.idx_first_sc, cfg.num_subcarriers
            )

            noise_power = np.mean(np.sum(np.abs(noise_spec) ** 2, axis=0))
            sample.snr_by_su[su_idx] = 10 * np.log10(signal_power / noise_power)

            total_spec += noise_spec

        # metrics describing how strongly the jammer stands out. In contrast to
        # the SJR above, they are not diluted by a jammer that only occupies a
        # small part of the time-frequency grid. They require both a jammer and
        # noise to be present, otherwise they are not defined.
        if len(sample.jammers) == 1 and noise:

            # noise power per resource element. It is estimated over the full
            # grid, so that the estimate is not distorted by the particular
            # noise realization on a potentially very small jammer footprint.
            noise_power_per_re = np.mean(np.abs(noise_spec) ** 2)

            # signal and jammer power per resource element, but restricted to
            # the resource elements the jammer occupies
            signal_power_local = np.mean(np.abs(signal_spec[jammer_mask]) ** 2)
            jammer_power_local = np.mean(np.abs(jammer_spec[jammer_mask]) ** 2)

            sample.jsnr_local_by_su[su_idx] = 10 * np.log10(
                jammer_power_local / (signal_power_local + noise_power_per_re)
            )

            # change of the spectrogram caused by the jammer, evaluated in the
            # logarithmic domain the spectrogram images are provided in. Single
            # resource elements can be attenuated by destructive interference,
            # which is intended and hence not clipped.
            db_contrast = 20 * np.log10(
                np.abs(total_spec) / np.abs(signal_spec + noise_spec)
            )

            sample.db_contrast_global_by_su[su_idx] = float(np.mean(db_contrast))
            sample.db_contrast_local_by_su[su_idx] = float(
                np.mean(db_contrast[jammer_mask])
            )
        else:
            sample.jsnr_local_by_su[su_idx] = np.nan
            sample.db_contrast_global_by_su[su_idx] = np.nan
            sample.db_contrast_local_by_su[su_idx] = np.nan

        spec_dBm = 20 * np.log10(np.abs(total_spec))

        sample.add_spectrogram(su_idx, spec_dBm)

    return sample


def add_localization_error(tx_pos, pos_std, scene_nr, scene_size):
    """Add a random error to the transmitter position.

    Parameters
    ----------
    tx_pos : list
        List containing x, y and z coordinates of the transmitter.
    pos_std : float
        Standard deviation of the position error.
    scene_nr : int
        Index of the current scene. Required to check whether the
        new location is inside an obstacle.
    scene_size : tuple
        Size of the scene.

    Returns
    -------
    tx_pos_est : list
        List containing the estimated x, y and z coordinates of the transmitter.
    """

    while True:
        # ensure the estimated transmitter position is not inside an obstacle
        r_err = np.random.normal(0, pos_std)
        phi_err = np.random.uniform(0, 2 * np.pi)
        x_err = r_err * np.cos(phi_err)
        y_err = r_err * np.sin(phi_err)
        tx_pos_est = [tx_pos[0] + x_err, tx_pos[1] + y_err, tx_pos[2]]

        if enclosed_in_obstacle(tx_pos_est, scene_nr):
            continue
        else:
            break

    # ensure, that the transmitter is not outside the room (otherwise, big
    # error in path loss due to wall)
    tx_pos_est[0] = np.clip(tx_pos_est[0], 0.05, scene_size[0] - 0.05)
    tx_pos_est[1] = np.clip(tx_pos_est[1], 0.05, scene_size[1] - 0.05)

    return tx_pos_est
