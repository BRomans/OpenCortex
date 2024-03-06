import pandas as pd
import numpy as np
from mne.io import RawArray
from mne import create_info
from mne.channels import make_standard_montage

unicorn_channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
unicorn_fs = 250


def load_data(path, header, fs, names=unicorn_channels, skiprows=5):
    if header:
        df = pd.read_csv(path,
                         names=names + ["trigger", "id", "target", "nontarget", "trial", "islast"],
                         skiprows=skiprows * fs)
        trigger = np.array(df.id)
    else:
        df = pd.read_csv(path, names=names + ["STI"], skiprows=skiprows * fs)
        trigger = np.array(df.STI)
    eeg = df.iloc[:, 0:len(unicorn_channels)].to_numpy()
    return eeg, trigger, df


def convert_to_mne(eeg, trigger, fs, chs, rescale=1e6, recompute=False):
    """
    Convert the data to MNE format
    :param eeg: numpy array of shape (n_samples, n_channels)
    :param trigger: numpy array of shape (n_samples, )
    :param fs: sampling frequency
    :param chs: list of channels names
    :param rescale: rescaling factor to the right units
    :param recompute: whether if changing trigger numerical values or not to avoid Event "0"
    :return: MNE RawArray object
    """

    this_rec = RawArray(eeg.T / rescale, create_info(chs, fs, ch_types='eeg'))

    # Get event indexes where value is not 0, i.e. -1 or 1
    pos = np.nonzero(trigger)[0]

    # Filter 0 values from the trigger array
    y = trigger[trigger != 0]

    # Create the stimuli channel
    stim_data = np.ones((1, this_rec.n_times)) if recompute else np.zeros((1, this_rec.n_times))

    # MNE works with absolute values of labels so -1 and +1 would result in only one kind of event
    # that's why we add 1 and obtain 1 and 2 as label values
    stim_data[0, pos] = (y + 1) if recompute else y

    stim_raw = RawArray(stim_data, create_info(['STI'], this_rec.info['sfreq'], ch_types=['stim']))

    # adding the stimuli channel (as a Raw object) to our EEG Raw object
    this_rec.add_channels([stim_raw])

    # Set the standard 10-20 montage
    montage = make_standard_montage('standard_1020')
    this_rec.set_montage(montage)
    return this_rec
