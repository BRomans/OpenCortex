"""
GUI Adapter - Connects StreamEngine service to PyQt GUI
"""

import logging
from PyQt5 import QtCore, QtWidgets
from opencortex.neuroengine.core.stream_engine import StreamEngine, StreamData
from typing import Dict, Any


class GUIAdapter(QtCore.QObject):
    """
    Adapter that connects StreamEngine service to PyQt GUI.
    Handles the translation between service callbacks and Qt signals.
    """

    # Qt signals for GUI updates
    data_updated = QtCore.pyqtSignal(object)  # StreamData object
    prediction_ready = QtCore.pyqtSignal(object)
    quality_updated = QtCore.pyqtSignal(list)
    classifier_ready = QtCore.pyqtSignal()
    training_complete = QtCore.pyqtSignal()
    error_occurred = QtCore.pyqtSignal(str)
    inference_mode_changed = QtCore.pyqtSignal(bool)
    trigger_sent = QtCore.pyqtSignal(int, float)

    def __init__(self, stream_engine: StreamEngine):
        super().__init__()
        self.stream_engine = stream_engine

        # Register with StreamEngine
        self.stream_engine.register_data_callback(self._on_data_update)
        self.stream_engine.register_event_callback(self._on_event)

        logging.info("GUI Adapter initialized")

    def _on_data_update(self, data: StreamData):
        """Handle data updates from StreamEngine"""
        self.data_updated.emit(data)
        self.quality_updated.emit(data.quality_scores)

    def _on_event(self, event_type: str, event_data: Dict[str, Any]):
        """Handle events from StreamEngine"""
        if event_type == 'prediction_ready':
            self.prediction_ready.emit(event_data['prediction'])
        elif event_type == 'classifier_ready':
            self.classifier_ready.emit()
        elif event_type == 'training_complete':
            self.training_complete.emit()
        elif event_type == 'error':
            self.error_occurred.emit(event_data['message'])
        elif event_type == 'inference_mode_changed':
            self.inference_mode_changed.emit(event_data['mode'])
        elif event_type == 'trigger_sent':
            self.trigger_sent.emit(event_data['trigger'], event_data['timestamp'])

    # =============== GUI TO ENGINE COMMANDS ===============

    def send_trigger(self, trigger: int):
        """Send trigger to StreamEngine"""
        self.stream_engine.send_command('send_trigger', {'trigger': trigger})

    def set_inference_mode(self, mode: bool = None):
        """Set inference mode"""
        self.stream_engine.send_command('set_inference_mode', {'mode': mode})

    def train_classifier(self):
        """Start classifier training"""
        self.stream_engine.send_command('train_classifier')

    def configure_filters(self, filter_config: Dict):
        """Configure data filters"""
        self.stream_engine.send_command('configure_filters', filter_config)

    def get_engine_status(self) -> Dict:
        """Get engine status"""
        return self.stream_engine.get_status()

    def cleanup(self):
        """Clean up adapter"""
        self.stream_engine.unregister_data_callback(self._on_data_update)
        self.stream_engine.unregister_event_callback(self._on_event)