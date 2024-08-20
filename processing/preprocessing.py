import logging
from mne import find_events, Epochs, create_info, EpochsArray
from mne.io import RawArray
from mne.time_frequency import psd_array_welch
import numpy as np


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


def extract_band_powers(data, fs, bands):
    """
    psd = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Channel 1
        [0.2, 0.3, 0.4, 0.5, 0.6],  # Channel 2
        [0.3, 0.4, 0.5, 0.6, 0.7],  # Channel 3
        [0.4, 0.5, 0.6, 0.7, 0.8]   # Channel 4
    ])

    band_power = np.sum(psd, axis=-1)
    # output: [1.5, 2.0, 2.5, 3.0]
    """
    psd, freqs = psd_array_welch(data=data, sfreq=fs, fmin=bands[0], fmax=bands[1], n_fft=fs * 2)
    band_power = np.sum(psd, axis=-1)
    return band_power
