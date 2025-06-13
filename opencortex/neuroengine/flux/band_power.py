import numpy as np

from opencortex.neuroengine.flux.base import Node
from opencortex.processing.preprocessing import extract_band_powers_fast as extract_band_powers
from opencortex.processing.proc_helper import freq_bands


class BandPowerExtractor(Node):
    """
        Extracts band powers from raw EEG data.
        Input shape: (channels, samples)
        Output: pandas.DataFrame
    """

    def __init__(self, fs: int, ch_names: list, name: str = None, average: bool = False, freq_bands: dict = freq_bands):
        super().__init__(name or "BandPowerExtractor")
        self.fs = fs
        self.ch_names = ch_names
        self.freq_bands = freq_bands
        self.average = False

    def __call__(self, data):
        return extract_band_powers(data=data, fs=self.fs, bands=self.freq_bands, average=self.average)

    def update_frequency_bands(self, freq_bands):
        """
        Update the frequency bands used for band power extraction.
        :param freq_bands: dict, new frequency bands
        """
        self.freq_bands = freq_bands

    def update_average(self, average: bool):
        """
        Update the averaging option for band power extraction.
        :param average: bool, whether to average the band powers across channels
        """
        self.average = average

def convert_bandpowers_for_lsl(band_powers):
    """Stack all bands into single array"""
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']


    arrays_to_stack = []
    for band_name in band_order:
        if band_name in band_powers:
            arrays_to_stack.append(band_powers[band_name])

    return np.concatenate(arrays_to_stack) if arrays_to_stack else np.array([])