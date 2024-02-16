import matplotlib
import matplotlib.pyplot as plt
from utils.loader import load_data, convert_to_mne, unicorn_channels
matplotlib.use("Qt5Agg")

unicorn_fs = 250

if __name__ == "__main__":
    eeg, trigger = load_data("../data/aep/auditory_erp_eyes_open_S1.csv", header=False, fs=unicorn_fs, skiprows=5)
    print("Loaded data with shape:" + str(eeg.shape) + " and trigger shape: " + str(trigger.shape))
    print("That means we have " + str(eeg.shape[0]) + " samples and " + str(eeg.shape[1]) + " channels.")

    # Convert to MNE format
    raw = convert_to_mne(eeg, trigger, fs=unicorn_fs, chs=unicorn_channels)

    # Compute PSD
    # fmax = Nyquist frequency, i.e. half of the sampling frequency
    # fmin = 0, i.e. the lowest frequency
    # you can adjust these values to zoom in on a specific frequency range
    Pxx = raw.compute_psd(fmin=0, fmax=unicorn_fs / 2)
    Pxx.plot()
    plt.show()

