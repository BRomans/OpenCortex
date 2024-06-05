from mne import find_events, Epochs
from mne.io import RawArray
import numpy as np


def basic_preprocessing_pipeline(data: RawArray, lp_freq: float = 1, hp_freq: float = 30, notch_freqs: tuple = (50, 60)):
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
    filtered = data.copy().notch_filter(freqs=notch_freqs[0])
    filtered = filtered.copy().notch_filter(freqs=notch_freqs[1])

    # Apply band-pass filtering
    # band-pass filter between 1 and 30 Hz
    filtered = filtered.copy().filter(l_freq=lp_freq, h_freq=hp_freq)

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


def extract_epochs(data: RawArray, events, ev_ids=None, tmin: float = -0.6, tmax: float = 0.8, baseline: tuple = (-.6, -.1)):
    """
    This function is used to extract epochs from the data.
    :param data: MNE RawArray object
    :param events: numpy array of shape (n_events, 3)
    :param ev_ids: dict, event ids
    :param tmin: float, minimum time
    :param tmax: float, maximum time
    :param baseline: tuple, baseline correction
    :return: MNE Epochs object
    """
    if ev_ids is None:
        epochs = Epochs(data, events=events, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    else:
        epochs = Epochs(data, events=events, event_id=ev_ids, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    return epochs
