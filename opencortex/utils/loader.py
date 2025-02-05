import pandas as pd
import numpy as np
from brainflow import BoardIds
from mne.io import RawArray
from mne import create_info
from mne.channels import make_standard_montage
from opencortex.utils.layouts import layouts


def load_data(path, header=None, fs=250, board=None, skiprows=5, delimiter=','):
    if board is None:
        df = pd.read_csv(path, skiprows=skiprows * fs, delimiter=delimiter)
        eeg = df.iloc[:, 0:-1].to_numpy()
        trigger = np.array(df.iloc[:, -1])
        return eeg, trigger, df

    start_eeg = layouts[board]["eeg_start"]
    end_eeg = layouts[board]["eeg_end"]
    df = pd.read_csv(path, names=layouts[board][header], skiprows=skiprows * fs, delimiter=delimiter)
    eeg = df.iloc[:, start_eeg:end_eeg].to_numpy()
    trigger = np.array(df.Trigger) if header != "unity" else np.array(df.id)

    return eeg, trigger, df


def load_erp_data(filepath, board_id, fs, chs, header, skiprows=5, delimiter='\t', start_id=98, end_id=99, session_length=60,
                  training=True):
    print(f'Board: {board_id}, fs: {fs}, channels: {chs}')

    eeg, trigger, dataframe = load_data(filepath, board=board_id,
                                        header=header, fs=fs, skiprows=skiprows, delimiter=delimiter)
    # infer n_classes from the trigger values but exclude start_id and end_id
    n_classes = len(np.unique(trigger[np.where((trigger < 90) & (trigger > 0))]))
    print(f'Found {n_classes} classes in the data')
    print(f'Unique trigger values: {np.unique(trigger[np.where((trigger < 90) & (trigger > 0))])}')

    # Check if train start and end triggers are present in the data
    if start_id not in trigger or end_id not in trigger:
        end_trigger = np.where(trigger == n_classes)[0][session_length - 1] if training else \
            np.where(trigger == n_classes)[0][-1]
    else:
        end_trigger = np.where(trigger == end_id)[0][0]

    first_trigger = np.where(trigger == start_id)[0][0] if len(np.where(trigger == end_id)[0]) > 0 else 0
    print(" Start index: " + str(first_trigger) + " End index: " + str(end_trigger))

    # Extract ERP data
    t_eeg = eeg[first_trigger:end_trigger + fs]
    t_trigger = trigger[first_trigger:end_trigger + fs]

    print("Loaded data with shape:" + str(t_eeg.shape) + " and trigger shape: " + str(t_trigger.shape))
    print("That means we have " + str(t_eeg.shape[0]) + " samples and " + str(t_eeg.shape[1]) + " channels.")
    return t_eeg, t_trigger


def convert_to_mne(eeg, trigger, fs, chs, rescale=1e6, recompute=False, transpose=True):
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
    eeg = eeg.T if transpose else eeg
    this_rec = RawArray(eeg / rescale, create_info(chs, fs, ch_types='eeg'))

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
