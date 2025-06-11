from opencortex.neuroengine.flux.base import Node
from opencortex.processing.preprocessing import extract_band_powers
from opencortex.processing.proc_helper import freq_bands


class BandPowerExtractor(Node):
    """
        Extracts band powers from raw EEG data.
        Input shape: (channels, samples)
        Output: pandas.DataFrame
    """

    def __init__(self, fs: int, ch_names: list, name: str = None):
        super().__init__(name or "BandPowerExtractor")
        self.fs = fs
        self.ch_names = ch_names

    def __call__(self, data):
        return extract_band_powers(data=data, fs=self.fs, bands=freq_bands, ch_names=self.ch_names)
