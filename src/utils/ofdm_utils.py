"""Contains utilities related to OFDM signal processing:

* Constellation mapping
* Resource allocation
* Creation of user and jammer signals
* Spectrogram calculation and processing
* Upsampling and filtering of resource grid and spectrograms
"""

__docformat__ = "numpy"

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as scsig
from sionna.phy.mapping import Mapper as ConstellationMapper
import seaborn as sns


def allocate_resources(num_tx, num_rb, num_slots, plot_allocation=False):
    """
    Assigns the resources (RBs) to the regular transmitters.

    Parameters
    ----------
    num_tx : int
        Number of transmitters.
    num_rb : int
        Number of RBs.
    num_slots : int
        Number of slots.
    plot_allocation : bool
        If True, the resource allocation is plotted.
        Default: False.

    Returns
    -------
    allocated_resources : dict
        Dictionary containing the allocated resources for each transmitter.
    """

    allocated_resources = {}

    # for plotting purposes
    total_allocated_resources = np.zeros((num_rb, num_slots))

    # the number of maximum RBs per transmitter is increased to not
    # have too many empty RBs
    max_rb_per_tx = (num_rb // num_tx) * 1.3

    # assign each transmitter a random number of RBs
    assigned_rb = np.random.randint(1, max_rb_per_tx + 1, num_tx)
    empty_rb = num_rb - np.sum(assigned_rb)

    # distribute the emmpty resource blocks
    if empty_rb > 0:

        # empty RBs per gap between two transmitters
        empty_rb_per_gap = [np.random.randint(0, empty_rb) for _ in range(num_tx + 1)]

        # calculate the scale factor to adjust the sum to the number of empty RBs
        if np.sum(empty_rb_per_gap) > empty_rb:
            scale_factor = empty_rb / np.sum(empty_rb_per_gap)
        else:
            scale_factor = 1

        # adjust the RBs that they sum up to empty_rb
        empty_rb_per_gap = [
            np.floor(value * scale_factor).astype(int) for value in empty_rb_per_gap
        ]

    else:
        empty_rb_per_gap = [0] * (num_tx + 1)

    start_idx = empty_rb_per_gap[0]
    for tx_idx in range(num_tx):
        allocated_resources[tx_idx] = np.zeros((num_rb, num_slots), dtype=bool)

        # generate random utilization in time
        utilization = np.random.uniform(0.3, 1)
        utilized_slots = np.random.binomial(1, utilization, num_slots)
        utilized_slots = np.where(utilized_slots == 1)[0]

        # make sure that at least one slot is utilized, otherwise there will
        # be nans in upsampling to REs
        if len(utilized_slots) == 0:
            utilized_slots = np.random.choice(num_slots, 1)

        allocated_resources[tx_idx][
            start_idx : start_idx + assigned_rb[tx_idx], utilized_slots
        ] = True

        if np.sum(allocated_resources[tx_idx]) == 0:
            print("Alloc err")

        # only for plotting update the total resource grid
        total_allocated_resources[
            start_idx : start_idx + assigned_rb[tx_idx], utilized_slots
        ] = (tx_idx + 1)

        start_idx += assigned_rb[tx_idx] + empty_rb_per_gap[tx_idx + 1]

    if plot_allocation:
        plt.figure()
        sns.heatmap(
            total_allocated_resources, cbar=True, cbar_kws={"label": "Transmitter ID"}
        )
        plt.xlabel("Time slot")
        plt.ylabel("RB")
        plt.title("Total resource allocation")
        plt.gca().invert_yaxis()
        plt.show()

    return allocated_resources


def upsample_rb_to_re(rb, n_subcarriers_per_rb, n_symbols_per_rb):
    """
    Upscales the resource blocks to the resource elements.

    Parameters
    ----------
    rb : np.ndarray
        Array of resource blocks.
    n_subcarriers_per_rb : int
        Number of subcarriers per resource block.
    n_symbols_per_rb : int
        Number of symbols per resource block (slot).

    Returns
    -------
    re : np.ndarray
        Array of resource elements.
    """

    # Get the shape of the original array
    orig_shape = rb.shape

    # Repeat elements along each axis
    re = np.repeat(
        np.repeat(rb, n_subcarriers_per_rb, axis=0), n_symbols_per_rb, axis=1
    )

    # Reshape the array to the new shape
    re = re.reshape(
        orig_shape[0] * n_subcarriers_per_rb, orig_shape[1] * n_symbols_per_rb
    )

    return re


def get_total_allocated_resources(
    sample, subcarriers_per_rb, symbols_per_slot, return_binary=True
):
    """Get the total allocated resources for all transmitters in the sample.

    Parameters
    -----
    sample : Sample
        Sample object containing the transmitters.
    subcarriers_per_rb : int
        Number of subcarriers per resource block.
    symbols_per_slot : int
        Number of symbols per slot.
    return_binary : bool
        If True, the returned array is binary, where True indicates an allocated resource.
        Otherwise, in the return array, 0 equals not allocated, and larger values indicate
        the corresponding transmitter index (starting with 1) that is allocated on the resource.
        Default: True.

    Returns
    -------
    total_allocated_resources : np.ndarray
        Total allocated resources for all transmitters in the sample
        in the shape [total_num_subcarriers, total_num_symbols].
    """
    # find all allocated resource  blocks for filtering
    total_allocated_resources = np.zeros_like(
        sample.transmitters[0].resources, dtype=np.int8
    )
    for tx_id, tx in enumerate(sample.transmitters):
        total_allocated_resources = total_allocated_resources + tx.resources * (
            tx_id + 1
        )

    if return_binary:
        total_allocated_resources = total_allocated_resources > 0

    total_allocated_resources = upsample_rb_to_re(
        total_allocated_resources, subcarriers_per_rb, symbols_per_slot
    )

    return total_allocated_resources


def generate_user_signal_freq(
    allocated_resources,
    sc_per_rb,
    sym_per_slot,
    bits_per_symbol=2,
):
    """Generate the signal for one user in the frequency domain,
    i.e., fill the time-frequency resource grid with QAM symbols.

    Parameters
    ----------
    allocated_resources : np.ndarray
        Array of allocated resource blocks.
    sc_per_rb : int
        Number of subcarriers per resource block.
    sym_per_slot : int
        Number of symbols per slot.
    bits_per_symbol : int
        Number of bits per symbol.
        Default: 2.

    Returns
    -------
    user_signal_freq : np.ndarray
        User signal in the frequency domain.
    """

    # derive number of bits needed from the number of allocated resource blocks
    n_bits = allocated_resources.sum() * sc_per_rb * sym_per_slot * bits_per_symbol

    bits_in = np.random.randint(0, 2, int(n_bits))

    mapper = ConstellationMapper("qam", bits_per_symbol)
    symbols = mapper(bits_in)

    # distribute the symbols on the allocated resource elements
    allocated_resource_elements = upsample_rb_to_re(
        allocated_resources, sc_per_rb, sym_per_slot
    )
    user_signal_freq = np.zeros_like(allocated_resource_elements, dtype=complex)
    user_signal_freq[allocated_resource_elements] = symbols

    return user_signal_freq


def freq_signal_to_time_signal(
    user_signal_freq, nfft, cp_len, idx_first_sc, add_cp=True
):
    """Conversion of the user (or jammer signal) from frequency to time domain.

    Assume, the user signal provided has N_SC subcarriers, then evenly zero
    subcarriers are padded to the left and right of the signal to match the FFT
    size NFFT. The signal is then converted to the time domain using the IFFT.

    No power scaling is applied.

    Parameters
    ----------
    user_signal_freq : np.ndarray
        User signal in frequency domain, e.g., allocated symbols in OFDM resource grid.
        It does not need to match the number of FFT points, as the system bandwidth can
        be smaller than the bandwidth simulated for reasonable number of fft points and
        sampling frequency.
    nfft : int
        FFT size.
    cp_len : int
        Length of the cyclic prefix in samples.
    idx_first_sc : int
        Index of the lowest subcarrier in the FFT that is potentially in use by the user
        signal.
    add_cp : bool
        If True, a cyclic prefix needs to be added when converting to time domain.
        Default: True.

    Returns
    -------
    time_signal : np.ndarray
        Time signal in the shape (nfft + cp_len) * num_symbols.
    """

    # consider full bandwidth of sampling and insert the considered resources
    full_bw_user_signal_freq = np.zeros(
        (nfft, user_signal_freq.shape[1]), dtype=complex
    )
    full_bw_user_signal_freq[
        idx_first_sc : idx_first_sc + user_signal_freq.shape[0], :
    ] = user_signal_freq

    time_signal = np.fft.ifft(
        np.fft.fftshift(full_bw_user_signal_freq.T, axes=1), n=nfft, axis=1
    )

    # append cyclic prefix
    if add_cp:
        time_signal = np.concatenate((time_signal[:, -cp_len:], time_signal), axis=1)

    time_signal = np.reshape(time_signal, -1)  # parallel to serial

    return time_signal


def generate_pilot_jammer_signal(
    allocated_rbs, subcarriers_per_rb, symbols_per_slot, pilot_slots
):
    """a pilot jammer jams the pilot symbols of one authorized transmitter.

    Assumptions: the symbol indexes of the pilots are known,
    the pilot symbols span every second SC in the specified symbol index

    Parameters
    ----------
    allocated_rbs : dict
        The allocated resources for each authorized user, whereby each dict entry
        represents the assigned resource blocks as boolean numpy array.
    subcarriers_per_rb : int
        Number of subcarriers per resource block.
    symbols_per_slot : int
        Number of symbols per slot.
    pilot_slots : list
        List of pilot symbol indexes in the slot.

    Returns
    -------
    jammer_signal_freq : np.ndarray
        Jammer signal in frequency domain.
    add_cp : bool
        If True, a cyclic prefix needs to be added when converting
        to time domain
    """

    jammed_tx_idx = np.random.choice(len(allocated_rbs))

    jammed_sc_low = np.where(allocated_rbs[jammed_tx_idx])[0].min() * subcarriers_per_rb
    jammed_sc_high = (
        np.where(allocated_rbs[jammed_tx_idx])[0].max() + 1
    ) * subcarriers_per_rb
    num_jammed_sc = int((jammed_sc_high - jammed_sc_low) / 2)
    assigned_slots = np.unique(np.where(allocated_rbs[jammed_tx_idx])[1])

    jammer_signal_freq = np.zeros_like(
        upsample_rb_to_re(
            allocated_rbs[jammed_tx_idx], subcarriers_per_rb, symbols_per_slot
        ),
        dtype=complex,
    )

    for slot_idx in assigned_slots:
        for pilot_idx in pilot_slots:
            jammer_signal_freq[
                jammed_sc_low:jammed_sc_high:2,
                slot_idx * symbols_per_slot + pilot_idx,
            ] = 1

    # add random phase
    jammer_signal_freq = jammer_signal_freq * np.exp(
        1j * np.random.uniform(0, 2 * np.pi, jammer_signal_freq.shape)
    )

    return jammer_signal_freq, True


def generate_barrage_jammer_signal(num_sc, num_symbols, nfft, cp_len):
    """A barrage jammer jams all subcarriers with white noise.

    Parameters
    ----------
    num_sc : int
        Number of subcarriers.
    num_symbols : int
        Number of symbols.
    nfft : int
        FFT size.
    cp_len : int
        Length of the cyclic prefix.

    Returns
    -------
    jammer_signal_freq : np.ndarray
        Jammer signal in frequency domain.
    add_cp : bool
        If True, a cyclic prefix needs to be added when converting
        to time domain.
    """

    num_symbols_jammer = num_symbols + int(np.ceil(cp_len * num_symbols / nfft))

    jammer_signal_freq = np.ones((num_sc, num_symbols_jammer)) * np.exp(
        2 * np.pi * 1j * np.random.uniform(0, 1, (num_sc, num_symbols_jammer))
    )

    return jammer_signal_freq, False


def generate_deceptive_jammer_signal(
    num_tx,
    num_rb,
    num_slots,
    sc_per_rb,
    sym_per_slot,
    bits_per_symbols=2,
):
    """
    Generate a deceptive jammer signal that pretends to be a legitimate user.

    Parameters
    ----------
    num_tx : int
        Number of transmitters.
    num_rb : int
        Number of resource blocks.
    num_slots : int
        Number of slots.
    sc_per_rb : int
        Number of subcarriers per resource block.
    sym_per_slot : int
        Number of symbols per slot.
    bits_per_symbols : int
        Number of bits per symbol.
        Default: 2.

    Returns
    -------
    jammer_signal_freq : np.ndarray
        Jammer signal in frequency domain.
    add_cp : bool
        If True, a cyclic prefix needs to be added when converting
        to time domain.
    """

    # randomly allocate resources to the jammer, but following the normal resource
    # allocation scheme
    allocated_resources = allocate_resources(num_tx, num_rb, num_slots)
    user_idx = np.random.choice(num_tx)

    n_bits = (
        allocated_resources[user_idx].sum()
        * sc_per_rb
        * sym_per_slot
        * bits_per_symbols
    )

    bits_in = np.random.randint(0, 2, int(n_bits))

    mapper = ConstellationMapper("qam", bits_per_symbols)
    symbols = mapper(bits_in)

    # distribute the symbols on the allocated resource elements
    allocated_resource_elements = upsample_rb_to_re(
        allocated_resources[user_idx], sc_per_rb, sym_per_slot
    )
    jammer_signal_freq = np.zeros_like(allocated_resource_elements, dtype=complex)
    jammer_signal_freq[allocated_resource_elements] = symbols

    return jammer_signal_freq, True


def generate_sweep_jammer_signal(num_sc, num_symbols, subcarriers_per_rb, nfft, cp_len):
    """Generate a sweep jammer signal which transmits narrowband noise with
    a sweeping center frequency.

    Parameters
    ----------
    num_sc : int
        Number of subcarriers.
    num_symbols : int
        Number of symbols.
    subcarriers_per_rb : int
        Number of subcarriers per resource block.
    nfft : int
        FFT size.
    cp_len : int
        Length of the cyclic prefix.

    Returns
    -------
    jammer_signal_freq : np.ndarray
        Jammer signal in frequency domain.
    add_cp : bool
        If True, a cyclic prefix needs to be added when converting
        to time domain.
    """

    # since no CP is added, more OFDM symbols need to be considered to create a time
    # signal of the same length as the user signal, which has CP
    num_symbols_jammer = num_symbols + int(np.ceil(cp_len * num_symbols / nfft))
    jammer_signal_freq = np.zeros((num_sc, num_symbols_jammer), dtype=complex)

    # number of instantaneous jammed subcarriers
    # assmuming minimum 1 SC and max 2 RB are jammed
    num_jammed_sc = np.random.randint(1, 2 * subcarriers_per_rb + 1)

    # step_sc specifies how many subcarriers the jamming frequency is increased with each symbol
    step_sc = np.random.randint(1, num_jammed_sc + 1)

    # choose max sweep duration in a way that ensures that there is no overflow,
    # but also maximum duration is half of the observed duration
    max_sweep_duration = min(
        (num_sc - num_jammed_sc) // step_sc,
        num_symbols_jammer / 2,
    )
    sweep_duration_symbols = np.random.randint(5, max_sweep_duration + 1)

    start_sc = np.random.randint(
        0,
        num_sc - (num_jammed_sc + step_sc * sweep_duration_symbols) + 1,
    )

    jammed_sc_current = start_sc
    for symbol_idx in range(jammer_signal_freq.shape[1]):
        jammer_signal_freq[
            jammed_sc_current : jammed_sc_current + num_jammed_sc, symbol_idx
        ] = 1
        if symbol_idx % sweep_duration_symbols == 0:
            jammed_sc_current = start_sc
        else:
            jammed_sc_current += step_sc

    # potentially flip the jamming pattern (decreasing jammer carrier frequency)
    if np.random.uniform() > 0.5:
        jammer_signal_freq = np.flip(jammer_signal_freq, axis=0)

    # add random phase
    jammer_signal_freq = jammer_signal_freq * np.exp(
        1j * np.random.uniform(0, 2 * np.pi, jammer_signal_freq.shape)
    )

    # rotate so that the sweep does not always start at the highest / lowest frequency
    jammer_signal_freq = np.roll(
        jammer_signal_freq, np.random.randint(0, num_symbols_jammer), axis=1
    )

    return jammer_signal_freq, False


def generate_random_hopping_tone_jammer_signal(num_sc, num_symbols, cp_len):
    """Generate a random hopping jammer signal which transmits a
    tone (sine signal) on a randomly hopping center frequency.
    Additionally, the jammer has a certain duty cycle, i.e., it is
    only active for a fraction of the time between the hops

    Parameters
    ----------
    num_sc : int
        Number of subcarriers.
    num_symbols : int
        Number of symbols.
    cp_len : int
        Length of the cyclic prefix.

    Returns
    -------
    jammer_signal_freq : np.ndarray
        Jammer signal in frequency domain.
    add_cp : bool
        If True, a cyclic prefix needs to be added when converting
        to time domain.
    """

    # the missing CP is compensated by adding additional OFDM symbols to match the
    # required length of the time signal
    num_symbols_jammer = num_symbols + int(np.ceil(cp_len * num_symbols / num_sc))

    hop_cycle = np.random.randint(2, num_symbols_jammer)
    duty_cycle = np.random.uniform(0.2, 1)
    duty_cycle_symbols = int(hop_cycle * duty_cycle)
    if duty_cycle_symbols == 0:
        # make sure the jammer is actually active
        duty_cycle_symbols = 1

    jammer_signal_freq = np.zeros((num_sc, num_symbols_jammer), dtype=complex)
    for symbol_idx in range(0, num_symbols_jammer, hop_cycle):
        jammed_sc = np.random.randint(0, num_sc)
        jammer_signal_freq[jammed_sc, symbol_idx : symbol_idx + duty_cycle_symbols] = 1

    # rotate so that the jammer might be also active at the beginning
    jammer_signal_freq = np.roll(
        jammer_signal_freq, np.random.randint(0, num_symbols_jammer), axis=1
    )

    return jammer_signal_freq, False


def generate_jammer_signal_freq(jammer_type, cfg, **kwargs):
    """Generate a jammer signal in the frequency domain.

    Parameters
    ----------
    jammer_type : str
        Type of the jammer.
    cfg : dict
        Configuration dictionary.
    (optional) allocated_resources : dict
        Dictionary containing the allocated resources for each
        authorized transmitter. Required for pilot jammer.
    (optional) num_tx : int
        Number of transmitters. Required for deceptive jammer.

    Returns
    -------
    jammer_signal : np.ndarray
        Jammer signal in frequency domain.
    add_cp : bool
        If True, a cyclic prefix needs to be added when converting
        to time domain.
    """

    if jammer_type == "barrage":
        jammer_signal, add_cp = generate_barrage_jammer_signal(
            cfg.num_subcarriers,
            cfg.n_slots * cfg.symbols_per_slot,
            cfg.nfft,
            cfg.cp_len,
        )
    elif jammer_type == "deceptive":
        if "num_tx" not in kwargs:
            raise ValueError("num_tx must be provided for deceptive jammer.")
        jammer_signal, add_cp = generate_deceptive_jammer_signal(
            kwargs["num_tx"],
            cfg.num_rb,
            cfg.n_slots,
            cfg.subcarriers_per_rb,
            cfg.symbols_per_slot,
            np.random.choice(cfg.bits_per_symbol),
        )
    elif jammer_type == "sweep":
        jammer_signal, add_cp = generate_sweep_jammer_signal(
            cfg.num_subcarriers,
            cfg.n_slots * cfg.symbols_per_slot,
            cfg.subcarriers_per_rb,
            cfg.nfft,
            cfg.cp_len,
        )
    elif jammer_type == "pilot":
        if "allocated_resources" not in kwargs:
            raise ValueError("allocated_resources must be provided for pilot jammer.")
        jammer_signal, add_cp = generate_pilot_jammer_signal(
            kwargs["allocated_resources"],
            cfg.subcarriers_per_rb,
            cfg.symbols_per_slot,
            cfg.pilot_slots,
        )
    elif jammer_type == "random_hop":
        jammer_signal, add_cp = generate_random_hopping_tone_jammer_signal(
            cfg.num_subcarriers, cfg.n_slots * cfg.symbols_per_slot, cfg.cp_len
        )
    else:
        raise ValueError(f"Jammer {jammer_type} type not implemented.")

    return jammer_signal, add_cp


def calc_complex_spectrogram(sig, nfft, hop, window="blackmanharris"):
    """Calculate the complex spectrogram of a signal.

    Parameters
    ----------
    sig : np.ndarray
        Signal to calculate the spectrogram from.
    nfft : int
        Number of FFT points.
    hop : int
        Shift of the window per step.
    window : str
        Window function to use for the spectrogram.

    Returns
    -------
    Sx : np.ndarray
        Complex spectrogram.
    """

    win = scsig.windows.get_window(window, nfft)
    win = win / np.sqrt(np.mean(win))  # normalize window to have unit energy

    spec = np.ones((nfft, np.ceil(len(sig) / hop).astype(int)), dtype=complex) * np.nan

    for idx in range(0, len(sig) - nfft, hop):
        spec[:, idx // hop] = np.fft.fft(
            sig[idx : idx + nfft] * win, n=nfft, norm="ortho"
        )

    if np.isnan(spec[0, -1]):
        # discard last column if it is not filled (if hop does not divide the signal length)
        spec = spec[:, :-1]

    spec = np.fft.fftshift(spec, axes=0)  # shift zero frequency to the center

    return spec


def calc_spectrogram(
    sig,
    nfft,
    hop,
    dynamic_range=50,
    window="blackmanharris",
    plot=True,
    plot_title="Spectrogram",
):
    """Calculate the spectrogram and optionally plot it.

    Parameters
    ----------
    sig : np.ndarray
        Signal to calculate the spectrogram from.
    nfft : int
        Number of FFT points.
    hop : int
        Shift of the window per step.
    fs : int
        Sampling frequency.
    dynamic_range : float
        Dynamic range in dB for clipping the spectrogram.
        Default: -50 dB.
        The low power values are clipped to this value below the maximum value.
        Should be only applied for plotting.
    window : str
        Window function to use for the spectrogram.
        Default: "blackmanharris".
        See `scipy.signal.windows` for available windows.
    plot : bool
        If True, the spectrogram is plotted.
        Default: True.
    plot_title : str
        Title of the plot.
        Default: "Spectrogram".

    Returns
    -------
    Sx_dB : np.ndarray
        Spectrogram (power spectral density) in dB.
    """

    spec = calc_complex_spectrogram(sig, nfft, hop, window)

    # convert to dBm
    spec_dBm = 20 * np.log10(np.abs(spec))

    spec_dBm = np.clip(spec_dBm, a_min=np.max(spec_dBm) - dynamic_range, a_max=None)

    if plot:
        fig, ax = plt.subplots()
        ax.set_title(plot_title)
        im = ax.imshow(
            spec_dBm,
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        fig.colorbar(im, label="Power Spectral Density [dB]")
        fig.tight_layout()
        plt.show()

    return spec_dBm


def crop_spectrogram_to_bandwidth(spec, idx_first_sc, num_sc):
    """Crop the spectrogram to the bandwidth of the signal.
    Spectrogram has the dimension (num_freq_bin, num_time_slots).

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram to crop.
    idx_first_sc : int
        Index of the first subcarrier in the spectrogram.
    num_sc : int
        Number of subcarriers to keep.

    Returns
    -------
    cropped_spec : np.ndarray
        Cropped spectrogram.
    """

    return spec[idx_first_sc : idx_first_sc + num_sc, :]


def filter_spectrogram_by_allocated_res(
    spec, total_allocated_resources, oob_suppression=-30, bounding_freq_only=False
):
    """Filter the spectrogram by the allocated resource elements.
    Resource elements that are not allocated are set to zero.

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram (in linear scale) to filter.
    total_allocated_resources : np.ndarray
        Array of allocated resources, where True indicates an allocated resource.
    oob_suppression : float
        Suppression level for out-of-band resource elements in dB.
        Default: -30 dB.
    bounding_freq_only : bool
        If True, not per subcarrier is filtered, but the bounding box is applied.
        This is required for  pilot jamming, where otherwise an extreme (and thereby unrealistic)
        suppression of single subcarriers would be ACHIEVED.

    Returns
    -------
    filtered_spec : np.ndarray
        Filtered spectrogram.
    """

    oob_suppression_linear = 10 ** (oob_suppression / 10)

    # create a mask which is True for the not allocated resource elements
    # which shall be suppressed.
    if bounding_freq_only:
        start_idx = np.where(total_allocated_resources)[0].min()
        end_idx = np.where(total_allocated_resources)[0].max() + 1
        mask = np.ones_like(total_allocated_resources, dtype=bool)
        mask[start_idx:end_idx] = False
    else:
        mask = ~total_allocated_resources.astype(bool)

    # apply suppression
    spec[mask] = spec[mask] * oob_suppression_linear

    return spec


def upsample_axis_of_2d_array(arr, new_dim, axis):
    """Upsample a 2D array along a specified axis using linear interpolation.

    Parameters
    ----------
    arr : np.ndarray
        2D array to upsample.
    new_dim : int
        New dimension size for the specified axis.
    axis : int
        Axis along which to upsample (0 or 1).

    Returns
    -------
    upsampled_array : np.ndarray
        Upsampled 2D array.
    """

    # upsample to actual number of time bins in the spectrogram
    x_old = np.linspace(0, 1, arr.shape[1])
    x_new = np.linspace(0, 1, new_dim)
    interp_func = interp1d(x_old, arr, kind="nearest", axis=axis)

    return interp_func(x_new)


def average_power_per_subcarrier(spec, allocated_resources_upsampled):
    """Calculate the average power per subcarrier in the spectrogram and
    replace the value on each subcarrier with the average power along the
    particular subcarrier along the time axis.

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram (in linear scale) to process.
    allocated_resources_upsampled : np.ndarray
        Array of allocated resources, where True indicates an allocated resource.

    Returns
    -------
    spec : np.ndarray
        Processed spectrogram with average power per subcarrier.
    """

    # Replace the value on each subcarrier with the average power
    for sc_idx in range(spec.shape[0]):
        if allocated_resources_upsampled[sc_idx].sum() > 0:
            allocated = spec[sc_idx, allocated_resources_upsampled[sc_idx].astype(bool)]
            average = np.power(
                10, np.mean(np.log10(np.abs(allocated)))
            )  # factor 20 cancels out
            spec[sc_idx, allocated_resources_upsampled[sc_idx].astype(bool)] = average

    return spec
