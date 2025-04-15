import logging
import numpy as np
from brainflow import DataFilter, FilterTypes, DetrendOperations
import pyqtgraph as pg
from PyQt5 import QtCore


class DataProcessor:
    def __init__(self, data_buffer, sampling_rate):
        self.data_buffer = data_buffer
        self.sampling_rate = sampling_rate

    def filter_data_buffer(self, eeg, channels, bandpass=False, notch=False, bp_low_fq=None, bp_high_fq=None, nt_freqs=None):
        for count, channel in enumerate(channels):
            ch_data = eeg[count]
            if bandpass and bp_low_fq is not None and bp_high_fq is not None:
                self.apply_bandpass_filter(ch_data, bp_low_fq, bp_high_fq)
            if notch and nt_freqs is not None:
                self.apply_notch_filter(ch_data, nt_freqs)

    def apply_bandpass_filter(self, ch_data, start_freq, stop_freq, order=4,
                              filter_type=FilterTypes.BUTTERWORTH_ZERO_PHASE, ripple=0):
        DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
        if start_freq >= stop_freq:
            logging.error("Band-pass Filter: Start frequency should be less than stop frequency")
            return
        if start_freq < 0 or stop_freq < 0:
            logging.error("Band-pass Filter: Frequency values should be positive")
            return
        if start_freq > self.sampling_rate / 2 or stop_freq > self.sampling_rate / 2:
            logging.error(
                "Band-pass Filter: Frequency values should be less than half of the sampling rate in respect of Nyquist theorem")
            return
        try:
            DataFilter.perform_bandpass(ch_data, self.sampling_rate, start_freq, stop_freq, order, filter_type, ripple)
        except ValueError as e:
            logging.error(f"Invalid frequency value {e}")

    def apply_notch_filter(self, ch_data, freqs, bandwidth=2.0, order=4, filter_type=FilterTypes.BUTTERWORTH_ZERO_PHASE,
                           ripple=0):
        for freq in freqs:
            if float(freq) < 0:
                logging.error("Frequency values should be positive")
                return
            if float(freq) > self.sampling_rate / 2:
                logging.error("Frequency values should be less than half of the sampling rate in respect of Nyquist "
                              "theorem")
                return
        try:
            for freq in freqs:
                start_freq = float(freq) - bandwidth
                end_freq = float(freq) + bandwidth
                DataFilter.perform_bandstop(ch_data, self.sampling_rate, start_freq, end_freq, order,
                                            filter_type, ripple)
        except ValueError:
            logging.error("Invalid frequency value")
