
import numpy as np

from opencortex.neuroengine.flux.base import Node


class QualityEstimator(Node):
    """
    Estimates signal quality for each EEG channel based on amplitude percentiles.
    Input shape: (channels, samples)
    Output: list of quality scores
    """

    def __init__(self, quality_thresholds, name: str = None):
        super().__init__(name or "QualityEstimator")
        self.quality_thresholds = quality_thresholds

    def __call__(self, data):
        # `data` is (channels, samples)
        return [self._estimate_quality(ch_data) for ch_data in data]

    def _estimate_quality(self, eeg, threshold=75):
        amplitude = np.percentile(eeg, threshold)
        for low, high, color, score in self.quality_thresholds:
            if low <= amplitude <= high:
                return score
        return 0  # default low quality