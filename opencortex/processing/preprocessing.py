import logging
from typing import Dict, Tuple, Optional, List, Union
from mne import find_events, Epochs, create_info, EpochsArray
from mne.io import RawArray
import numpy as np
from numpy import trapz
from scipy.signal import welch, periodogram
from scipy.integrate import simpson


def basic_preprocessing_pipeline(data: RawArray, lp_freq: float = 1, hp_freq: float = 30, notch_freqs: tuple = (50, 60),
                                 filter_length='auto'):
    """
    This function is used to do basic preprocessing on the data.
    :param data: MNE RawArray object
    :param lp_freq: float, low-pass frequency
    :param hp_freq: float, high-pass frequency
    :param notch_freqs: tuple, notch filter frequencies
    :return: MNE RawArray object
    """
    # Apply notch filter
    # notch filter at 50 Hz and 60 Hz
    filtered = data.copy().notch_filter(freqs=[notch_freqs[0], notch_freqs[1]], filter_length=filter_length,
                                        trans_bandwidth=7.0)

    # Apply band-pass filtering
    # band-pass filter
    filtered = filtered.copy().filter(l_freq=lp_freq, h_freq=hp_freq, filter_length=filter_length,
                                      l_trans_bandwidth=1.0, h_trans_bandwidth=3.0)

    return filtered


def extract_events(data: RawArray, stim_channel='STI', ev_ids=None, event_color=None):
    """
    This function is used to extract events from the data.
    :param data: MNE RawArray object
    :param stim_channel: str, stimulus channel name
    :param ev_ids: dict, event ids
    :param event_color: dict, event colors
    :return: numpy array of shape (n_events, 3), dict, dict
    """
    if ev_ids is None:
        ev_ids = {'NT': 1}
    if event_color is None:
        event_color = {1: 'r'}

    events = find_events(data, stim_channel=stim_channel)

    labels = np.unique(events[:, 2])
    for i in range(1, len(labels)):
        ev_ids['T' + str(i)] = i + 1
        event_color[i + 1] = 'g'
    return events, ev_ids, event_color


def extract_epochs(data: RawArray, events, ev_ids=None, reject=None, tmin: float = -0.6, tmax: float = 0.8,
                   baseline: tuple = (-.6, -.1)):
    """
    This function is used to extract epochs from the data.
    :param data: MNE RawArray object
    :param events: numpy array of shape (n_events, 3)
    :param ev_ids: dict, event ids
    :param reject: dict, rejection parameters
    :param tmin: float, minimum time
    :param tmax: float, maximum time
    :param baseline: tuple, baseline correction
    :return: MNE Epochs object
    """
    try:
        if ev_ids is None:
            epochs = Epochs(data, events=events, reject=reject, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
        else:
            epochs = Epochs(data, events=events, event_id=ev_ids, reject=reject, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True)
        return epochs
    except ValueError:
        logging.error("All epochs were dropped. Please check the rejection parameters.")


def make_overlapping_epochs(data: RawArray, events, tmin: float = -0.1, tmax: float = 0.5, baseline=None,
                            fs: int = 250):
    pre_event_samples = int(-tmin * fs)
    post_event_samples = int(tmax * fs)
    # Extract epochs manually
    epochs_data = []
    for event in events:
        start_sample = event[0] - pre_event_samples
        end_sample = event[0] + post_event_samples
        epoch = data[:, start_sample:end_sample][0]
        logging.debug(f"Epoch shape: {epoch.shape}")
        epochs_data.append(epoch)
    epochs_data = np.array(epochs_data)
    logging.debug(f"Epochs shape: {epochs_data.shape}")

    # Create an MNE EpochsArray
    info = create_info(ch_names=data.ch_names, sfreq=fs, ch_types='eeg')
    return EpochsArray(data=epochs_data, info=info, events=events, tmin=tmin, baseline=baseline)


def extract_band_powers_improved(
        data: np.ndarray,
        fs: float,
        bands: Dict[str, Tuple[float, float]],
        ch_names: Optional[List[str]] = None,
        average: bool = False,
        method: str = 'welch',
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        window: str = 'hann',
        detrend: str = 'constant',
        return_freqs: bool = False,
        normalize: bool = False,
        relative: bool = False,
        integration_method: str = 'trapz'
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    Enhanced band power extraction with global average setting.

    Parameters:
    -----------
    data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    bands : dict
        Band definitions: {'alpha': (8, 12), 'beta': (12, 30), ...}
    ch_names : list, optional
        Channel names. If None, uses ['Ch1', 'Ch2', ...]
    average : bool
        If True, compute mean across channels. If False, return per-channel values
    method : str
        PSD method: 'welch', 'periodogram'
    nperseg : int, optional
        Length of each segment for Welch method. Default: fs (1 second)
    noverlap : int, optional
        Overlap between segments for Welch method. Default: nperseg // 2
    window : str
        Window function for PSD computation
    detrend : str
        Detrending method: 'constant', 'linear', or False
    return_freqs : bool
        If True, also return frequency array
    normalize : bool
        If True, normalize PSD by total power
    relative : bool
        If True, return relative band powers (as percentage of total power)
    integration_method : str
        Method for power integration: 'trapz', 'simpson', 'sum'

    Returns:
    --------
    band_powers : dict
        Dictionary with band names as keys and power arrays as values
    freqs : np.ndarray (optional)
        Frequency array if return_freqs=True
    """

    # Validate inputs
    if data.ndim != 2:
        raise ValueError("Data must be 2D array with shape (n_channels, n_samples)")

    n_channels, n_samples = data.shape

    # Set default channel names
    if ch_names is None:
        ch_names = [f'Ch{i + 1}' for i in range(n_channels)]
    elif len(ch_names) != n_channels:
        raise ValueError(f"Number of channel names ({len(ch_names)}) must match number of channels ({n_channels})")

    # Set default parameters
    if nperseg is None:
        nperseg = min(int(fs), n_samples)  # 1 second or available data
    if noverlap is None:
        noverlap = nperseg // 2

    # Validate bands
    for band_name, (fmin, fmax) in bands.items():
        if fmin >= fmax:
            raise ValueError(f"Invalid band '{band_name}': fmin ({fmin}) must be < fmax ({fmax})")
        if fmax > fs / 2:
            logging.warning(f"Band '{band_name}' fmax ({fmax}) exceeds Nyquist frequency ({fs / 2})")

    # Compute PSD once for all channels
    logging.debug(f"Computing PSD using {method} method")

    if method == 'welch':
        freqs, psd = welch(
            data, fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            axis=-1  # Compute along time axis
        )
    elif method == 'periodogram':
        freqs, psd = periodogram(
            data, fs,
            window=window,
            detrend=detrend,
            axis=-1
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'welch' or 'periodogram'")

    # Normalize PSD if requested
    if normalize:
        psd_total = np.sum(psd, axis=-1, keepdims=True)
        psd = psd / psd_total

    # Extract band powers
    band_powers = {}
    freq_res = freqs[1] - freqs[0]  # Frequency resolution

    for band_name, (fmin, fmax) in bands.items():
        # Find frequency indices for this band
        idx_band = (freqs >= fmin) & (freqs <= fmax)

        if not np.any(idx_band):
            logging.warning(f"No frequency bins found for band '{band_name}' ({fmin}-{fmax} Hz)")
            band_powers[band_name] = np.zeros(n_channels)
            continue

        # Extract PSD for this band
        psd_band = psd[:, idx_band]  # Shape: (n_channels, n_freq_bins)
        freqs_band = freqs[idx_band]

        # Compute band power using specified integration method
        if integration_method == 'trapz':
            # Trapezoidal integration (most accurate)
            band_power = np.array([trapz(psd_band[ch, :], freqs_band) for ch in range(n_channels)])
        elif integration_method == 'simpson':
            # Simpson's rule (requires scipy)
            band_power = np.array([simpson(psd_band[ch, :], freqs_band) for ch in range(n_channels)])
        elif integration_method == 'sum':
            # Simple summation
            band_power = np.sum(psd_band, axis=-1) * freq_res
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")

        band_powers[band_name] = band_power

    # Compute relative powers if requested
    if relative:
        # Compute total power across all frequencies
        total_power = np.array([trapz(psd[ch, :], freqs) for ch in range(n_channels)])

        for band_name in band_powers:
            band_powers[band_name] = (band_powers[band_name] / total_power) * 100  # As percentage

    # Apply global averaging if requested
    if average:
        band_powers = {band: np.mean(power) for band, power in band_powers.items()}
        logging.debug("Applied global averaging across channels")

    # Log results
    logging.debug(f"Extracted band powers for {len(bands)} bands across {n_channels} channels")

    if return_freqs:
        return band_powers, freqs
    else:
        return band_powers


def extract_band_powers_fast(
        data: np.ndarray,
        fs: float,
        bands: Dict[str, Tuple[float, float]],
        average: bool = False,
        nperseg: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Fast version with minimal parameters for real-time use.

    This is a streamlined version of your original function with key improvements.
    """

    n_channels, n_samples = data.shape

    # Set default nperseg
    if nperseg is None:
        nperseg = min(int(fs), n_samples)

    # Compute PSD once for all channels using Welch method
    freqs, psd = welch(data, fs, nperseg=nperseg, axis=-1)
    freq_res = freqs[1] - freqs[0]

    # Extract band powers
    band_powers = {}

    for band_name, (fmin, fmax) in bands.items():
        if fmin >= fmax:
            raise ValueError(f"Invalid band: {band_name}")

        # Find frequency indices
        idx_band = (freqs >= fmin) & (freqs <= fmax)

        if np.any(idx_band):
            # Sum power in frequency band (multiply by freq_res for proper integration)
            band_power = np.sum(psd[:, idx_band], axis=-1) * freq_res
        else:
            band_power = np.zeros(n_channels)

        band_powers[band_name] = band_power

    # Apply global averaging if requested
    if average:
        band_powers = {band: np.mean(power) for band, power in band_powers.items()}

    return band_powers


def extract_band_powers_optimized(
        data: np.ndarray,
        fs: float,
        bands: Dict[str, Tuple[float, float]],
        average: bool = False,
        psd_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Memory-optimized version that can reuse PSD computation.

    Returns both band powers and PSD cache for reuse.
    """

    # Use cached PSD if available, otherwise compute
    if psd_cache is not None:
        freqs, psd = psd_cache
        logging.debug("Using cached PSD")
    else:
        freqs, psd = welch(data, fs, nperseg=min(int(fs), data.shape[1]), axis=-1)
        logging.debug("Computed new PSD")

    freq_res = freqs[1] - freqs[0]
    band_powers = {}

    # Vectorized band power extraction
    for band_name, (fmin, fmax) in bands.items():
        if fmin >= fmax:
            raise ValueError(f"Invalid band: {band_name}")

        idx_band = (freqs >= fmin) & (freqs <= fmax)

        if np.any(idx_band):
            band_power = np.sum(psd[:, idx_band], axis=-1) * freq_res
        else:
            band_power = np.zeros(psd.shape[0])

        band_powers[band_name] = band_power

    if average:
        band_powers = {band: np.mean(power) for band, power in band_powers.items()}

    return band_powers, (freqs, psd)


# ================== USAGE EXAMPLES ==================

def example_usage():
    """Examples of how to use the improved functions"""

    # Generate sample data
    fs = 250  # Sampling rate
    n_channels = 16
    n_samples = 1000
    data = np.random.randn(n_channels, n_samples)

    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 50)
    }

    # Example 1: Basic usage (replacement for your original function)
    band_powers = extract_band_powers_fast(data, fs, bands, average=False)
    print(f"Band powers shape: {[bp.shape for bp in band_powers.values()]}")

    # Example 2: Advanced usage with all features
    band_powers, freqs = extract_band_powers_improved(
        data, fs, bands,
        ch_names=[f'EEG_{i:02d}' for i in range(n_channels)],
        average=False,
        method='welch',
        return_freqs=True,
        relative=False,
        integration_method='trapz'
    )
    print(f"Advanced: {len(band_powers)} bands, freq range: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz")

    # Example 3: Global averaging
    avg_powers = extract_band_powers_fast(data, fs, bands, average=True)
    print(f"Averaged powers: {avg_powers}")

    # Example 4: Relative powers
    rel_powers = extract_band_powers_improved(data, fs, bands, relative=True)
    print(f"Relative powers (% of total): {[np.mean(bp) for bp in rel_powers.values()]}")
