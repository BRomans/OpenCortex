import matplotlib
import matplotlib.pyplot as plt
from brainflow import BoardIds, BoardShim

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.layouts import layouts
from utils.loader import load_data, convert_to_mne
matplotlib.use("Qt5Agg")

board_id = BoardIds.UNICORN_BOARD
fs = BoardShim.get_sampling_rate(board_id)
chs = layouts[board_id]["channels"]

if __name__ == "__main__":
    eeg, trigger, _ = load_data("../data/aep/auditory_erp_eyes_open_S1.csv", header=False, fs=fs, skiprows=5)
    print("Loaded data with shape:" + str(eeg.shape) + " and trigger shape: " + str(trigger.shape))
    print("That means we have " + str(eeg.shape[0]) + " samples and " + str(eeg.shape[1]) + " channels.")

    # Convert to MNE format
    raw = convert_to_mne(eeg, trigger, fs=fs, chs=chs, recompute=False, transpose=True)

    # Compute PSD
    # fmax = Nyquist frequency, i.e. half of the sampling frequency
    # fmin = 0, i.e. the lowest frequency
    # you can adjust these values to zoom in on a specific frequency range
    Pxx = raw.compute_psd(fmin=0, fmax=fs / 2)
    Pxx.plot()
    plt.show()

