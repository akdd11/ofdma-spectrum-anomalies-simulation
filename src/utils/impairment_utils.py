"""Hardware impairment models for sensing units (SUs).

Impairments implemented here:

* SU STFT timing offset against the OFDM symbol grid
* SU LO frequency offset, common to all emitters at that SU
* RX DC offset
* RX IQ imbalance

Hardware parameters are properties of a device, not of a sample: they are
drawn once per SU with a dedicated seed via `generate_su_hardware_params` and
held fixed for the whole dataset. The ``apply_*`` functions below are pure
transformations applied per sample using those fixed parameters.
"""

__docformat__ = "numpy"

import numpy as np
import pandas as pd


def generate_su_hardware_params(num_su, cfg):
    """Draw fixed per-SU hardware impairment parameters.

    Parameters are drawn once for the whole dataset using a dedicated seed
    (``cfg.impairments.su_hardware_seed``), independent of which impairments are
    currently enabled, so that toggling one impairment on/off does not change
    the draws of the others.

    Parameters
    ----------
    num_su : int
        Number of sensing units.
    cfg : OmegaConf
        Configuration object, must contain the ``impairments`` block (see
        ``dataset_generation.yaml``).

    Returns
    -------
    su_hardware : pd.DataFrame
        One row per SU (indexed by ``su_idx``), with columns:
        ``timing_offset_samples``, ``lo_freq_offset_ppm``,
        ``dc_offset_over_noise_db``, ``dc_offset_phase_rad``,
        ``iq_gain_imbalance_db``, ``iq_phase_imbalance_deg``, ``iq_irr_db``
        (derived, informational only).
    """

    rng = np.random.default_rng(cfg.impairments.su_hardware_seed)

    # The SU is time-synchronized to the network to 5G-UE accuracy, so
    # its FFT-window timing error is only a small *residual* about the ideal
    # symbol boundary, not a uniform draw over a full symbol. Modelled as
    # |residual| in [0, max] samples, where max comes from `max_timing_error_ns`
    # (default 390 ns = 3GPP TS 38.133 Te for 15 kHz SCS ~ 12 samples at
    # 30.72 MHz, ~8 % of the CP). Because it stays inside the ISI-free part of
    # the CP its magnitude-spectrogram effect is a pure phase ramp (negligible).
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
            f"({cfg.cp_len}); the synchronized-SU model assumes a sub-CP residual."
        )
    timing_offset_samples = rng.integers(0, max_timing_error_samples + 1, size=num_su)

    # The SU is also frequency-locked to the network DL, so its LO offset is a
    # residual about the network carrier (default <= 0.1 ppm, the 3GPP UE
    # frequency-error limit; 375 Hz = 0.025 SCS at 3.75 GHz), not the
    # tolerance of a free-running TCXO. Signed, magnitude uniform in the range.
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

    # informational only: IRR resulting from the drawn gain/phase imbalance
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
        Complex exponential of length ``num_samples``, to be multiplied
        elementwise with the (already channel-convolved) received time
        signal, before `apply_iq_imbalance`.
    """

    t = np.arange(num_samples) / sample_rate_hz
    return np.exp(-1j * 2 * np.pi * delta_f_hz * t)


def compute_iq_imbalance_coeffs(gain_imbalance_db, phase_imbalance_deg):
    """Compute the alpha/beta coefficients of a frequency-flat IQ imbalance.

    Uses the standard model ``y = alpha * x + beta * conj(x)``.

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


def apply_iq_imbalance(sig, alpha, beta):
    """Apply a frequency-flat IQ imbalance to a complex baseband time signal.

    Applied before the STFT, so the conjugate image of subcarrier k lands
    exactly on -k. Since the STFT window
    is real, this is equivalent to ``alpha * S[k] + beta * conj(S[-k])`` in the
    spectrogram domain, but needs no index bookkeeping on the cropped grid.

    Parameters
    ----------
    sig : np.ndarray
        Complex baseband time signal (after channel and LO rotation).
    alpha, beta : complex
        Coefficients from `compute_iq_imbalance_coeffs`.

    Returns
    -------
    sig_out : np.ndarray
        IQ-imbalanced time signal, same shape as ``sig``.
    """

    return alpha * sig + beta * np.conj(sig)


def apply_dc_offset(noise_sig, nfft, offset_over_noise_db, phase_rad):
    """Add a persistent per-SU DC offset to a complex baseband noise signal.

    Folded into the noise/background component rather than any signal or
    jammer component, since the DC offset belongs to neither. Added as a
    constant in the time domain, so the STFT window spreads it over the bins
    around DC exactly as in a real receiver (7 bins for Blackman-Harris).

    The magnitude is parameterized relative to the actual noise power of the
    given ``noise_sig``, so it tracks the instantaneous noise realization while
    the dB offset and phase stay fixed hardware properties of the SU.
    ``offset_over_noise_db`` is the *total* DC power relative to the noise power
    per FFT bin (``mean(|noise_sig|^2) / nfft``). In the spectrogram, the sum of
    the DC footprint over all bins therefore sits ``offset_over_noise_db`` above
    the mean per-RE noise power; the centre bin alone holds 1/ENBW of it
    (about -3 dB for Blackman-Harris).

    Parameters
    ----------
    noise_sig : np.ndarray
        Complex baseband noise time signal.
    nfft : int
        FFT size of the subsequent STFT, i.e. the number of bins the noise
        power is spread over.
    offset_over_noise_db : float
        Total DC offset power relative to the noise power per FFT bin, in dB.
    phase_rad : float
        Fixed phase of the (complex) DC offset.

    Returns
    -------
    noise_sig_out : np.ndarray
        ``noise_sig`` plus the constant DC offset.
    """

    noise_power_per_bin = np.mean(np.abs(noise_sig) ** 2) / nfft
    dc_amplitude = np.sqrt(noise_power_per_bin * 10 ** (offset_over_noise_db / 10))
    return noise_sig + dc_amplitude * np.exp(1j * phase_rad)
