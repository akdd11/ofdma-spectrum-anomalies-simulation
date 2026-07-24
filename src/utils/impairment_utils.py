"""Hardware impairment models for sensing units (SUs).

See `hardware_impairments.md` (repository root) for the physical rationale,
literature references and parameterization decisions behind every impairment
implemented here:

* SU STFT timing offset against the OFDM symbol grid (§4.3)
* SU LO frequency offset, common to all emitters at that SU (§4.4)
* RX DC offset (§4.1)
* RX IQ imbalance (§4.2)

Hardware parameters are properties of a device, not of a sample (D2): they are
drawn once per SU with a dedicated seed via `generate_su_hardware_params` and
held fixed for the whole dataset. The `apply_*` functions below are pure
transformations applied per sample using those fixed parameters.
"""

__docformat__ = "numpy"

import numpy as np
import pandas as pd


def generate_su_hardware_params(num_su, cfg):
    """Draw fixed per-SU hardware impairment parameters.

    Parameters are drawn once for the whole dataset using a dedicated seed
    (`cfg.impairments.su_hardware_seed`), independent of which impairments are
    currently enabled, so that toggling one impairment on/off does not change
    the draws of the others.

    Parameters
    ----------
    num_su : int
        Number of sensing units.
    cfg : OmegaConf
        Configuration object, must contain the `impairments` block (see
        `dataset_generation.yaml`).

    Returns
    -------
    su_hardware : pd.DataFrame
        One row per SU (indexed by `su_idx`), with columns:
        `timing_offset_samples`, `lo_freq_offset_ppm`,
        `dc_offset_over_noise_db`, `dc_offset_phase_rad`,
        `iq_gain_imbalance_db`, `iq_phase_imbalance_deg`, `iq_irr_db`
        (derived, informational only).
    """

    rng = np.random.default_rng(cfg.impairments.su_hardware_seed)

    # §4.3: the SU is time-synchronized to the network to 5G-UE accuracy, so
    # its FFT-window timing error is only a small *residual* about the ideal
    # symbol boundary, not a uniform draw over a full symbol. Modelled as
    # |residual| in [0, max] samples, where max comes from `max_timing_error_ns`
    # (default 390 ns = 3GPP TS 38.133 Te for 15 kHz SCS ~ 12 samples at
    # 30.72 MHz, ~8 % of the CP). Because it stays inside the ISI-free part of
    # the CP its magnitude-spectrogram effect is a pure phase ramp (negligible);
    # see hardware_impairments.md §4.3.
    sample_rate_hz = cfg.subcarrier_spacing * cfg.nfft
    max_timing_error_samples = int(
        round(
            cfg.impairments.su_timing_offset.max_timing_error_ns * 1e-9 * sample_rate_hz
        )
    )
    if max_timing_error_samples > cfg.cp_len:
        raise ValueError(
            "impairments.su_timing_offset.max_timing_error_ns corresponds to "
            f"{max_timing_error_samples} samples, which exceeds the cyclic prefix "
            f"({cfg.cp_len}); the synchronized-SU model assumes a sub-CP residual "
            "(see hardware_impairments.md §4.3)."
        )
    timing_offset_samples = rng.integers(0, max_timing_error_samples + 1, size=num_su)

    lo_ppm_magnitude = rng.uniform(
        cfg.impairments.su_lo_freq_offset.ppm_min,
        cfg.impairments.su_lo_freq_offset.ppm_max,
        size=num_su,
    )
    lo_sign = rng.choice([-1.0, 1.0], size=num_su)
    lo_freq_offset_ppm = lo_sign * lo_ppm_magnitude

    dc_offset_over_noise_db = rng.uniform(
        cfg.impairments.rx_dc_offset.offset_over_noise_db_min,
        cfg.impairments.rx_dc_offset.offset_over_noise_db_max,
        size=num_su,
    )
    dc_offset_phase_rad = rng.uniform(0, 2 * np.pi, size=num_su)

    iq_gain_imbalance_db = rng.uniform(
        cfg.impairments.rx_iq_imbalance.gain_imbalance_db_min,
        cfg.impairments.rx_iq_imbalance.gain_imbalance_db_max,
        size=num_su,
    )
    iq_phase_imbalance_deg = rng.uniform(
        cfg.impairments.rx_iq_imbalance.phase_imbalance_deg_min,
        cfg.impairments.rx_iq_imbalance.phase_imbalance_deg_max,
        size=num_su,
    )

    # informational only: IRR resulting from the drawn gain/phase imbalance,
    # see hardware_impairments.md §4.2 for the derivation of this expression
    g = 10 ** (iq_gain_imbalance_db / 20)
    phi = np.deg2rad(iq_phase_imbalance_deg)
    iq_irr_db = 10 * np.log10(
        (1 + g**2 + 2 * g * np.cos(phi)) / (1 + g**2 - 2 * g * np.cos(phi))
    )

    su_hardware = pd.DataFrame(
        {
            "su_idx": np.arange(num_su),
            "timing_offset_samples": timing_offset_samples,
            "lo_freq_offset_ppm": lo_freq_offset_ppm,
            "dc_offset_over_noise_db": dc_offset_over_noise_db,
            "dc_offset_phase_rad": dc_offset_phase_rad,
            "iq_gain_imbalance_db": iq_gain_imbalance_db,
            "iq_phase_imbalance_deg": iq_phase_imbalance_deg,
            "iq_irr_db": iq_irr_db,
        }
    ).set_index("su_idx")

    return su_hardware


def lo_frequency_rotation(delta_f_hz, sample_rate_hz, num_samples):
    """Complex baseband rotation representing a receiver LO frequency offset.

    Parameters
    ----------
    delta_f_hz : float
        Frequency offset in Hz (signed).
    sample_rate_hz : float
        Sampling rate in Hz.
    num_samples : int
        Number of time-domain samples to generate the rotation for.

    Returns
    -------
    rotation : np.ndarray
        Complex exponential of length `num_samples`, to be multiplied
        elementwise with the (already channel-convolved) received time
        signal.
    """

    t = np.arange(num_samples) / sample_rate_hz
    return np.exp(-1j * 2 * np.pi * delta_f_hz * t)


def compute_iq_imbalance_coeffs(gain_imbalance_db, phase_imbalance_deg):
    """Compute the alpha/beta coefficients of a frequency-flat IQ imbalance.

    Uses the standard model `y = alpha * x + beta * conj(x)`, see
    hardware_impairments.md §4.2.

    Parameters
    ----------
    gain_imbalance_db : float
        Gain mismatch between the I and Q paths, in dB.
    phase_imbalance_deg : float
        Phase mismatch between the I and Q paths, in degrees.

    Returns
    -------
    alpha, beta : complex
        Coefficients of the IQ imbalance model.
    """

    g = 10 ** (gain_imbalance_db / 20)
    phi = np.deg2rad(phase_imbalance_deg)
    alpha = (1 + g * np.exp(1j * phi)) / 2
    beta = (1 - g * np.exp(-1j * phi)) / 2
    return alpha, beta


def apply_iq_imbalance(spec, alpha, beta):
    """Apply a frequency-flat IQ imbalance to a cropped complex spectrogram.

    Mirrors the spectrogram about its own centre row, which is equivalent to
    mirroring about DC because the crop is centred on the FFT's DC bin (see
    hardware_impairments.md §4.2).

    Parameters
    ----------
    spec : np.ndarray
        Complex spectrogram, shape (num_subcarriers, num_symbols).
    alpha, beta : complex
        Coefficients from `compute_iq_imbalance_coeffs`.

    Returns
    -------
    spec_out : np.ndarray
        IQ-imbalanced spectrogram, same shape as `spec`.
    """

    return alpha * spec + beta * np.conj(spec[::-1, :])


def apply_dc_offset(noise_spec, dc_bin_idx, offset_over_noise_db, phase_rad):
    """Add a persistent per-SU DC offset to the centre (DC) bin of a spectrogram.

    Folded into the noise/background component (D4) rather than any signal or
    jammer component, since the DC offset belongs to neither. The magnitude
    is parameterized relative to the actual per-RE noise floor of the given
    `noise_spec` (see hardware_impairments.md §4.1), so it tracks the
    instantaneous noise realization while the dB offset and phase stay fixed
    hardware properties of the SU.

    Parameters
    ----------
    noise_spec : np.ndarray
        Complex noise spectrogram, shape (num_subcarriers, num_symbols).
        Modified in place and also returned.
    dc_bin_idx : int
        Row index of the DC bin in the cropped spectrogram grid
        (`num_subcarriers // 2`).
    offset_over_noise_db : float
        DC offset power relative to the mean per-RE noise power, in dB.
    phase_rad : float
        Fixed phase of the (complex) DC offset.

    Returns
    -------
    noise_spec : np.ndarray
        `noise_spec` with the DC offset added at `dc_bin_idx`, for every
        column.
    """

    noise_power_per_re = np.mean(np.abs(noise_spec) ** 2)
    dc_amplitude = np.sqrt(noise_power_per_re * 10 ** (offset_over_noise_db / 10))
    noise_spec[dc_bin_idx, :] += dc_amplitude * np.exp(1j * phase_rad)
    return noise_spec
