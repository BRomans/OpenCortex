"""
This class creates a dialog to select the EEG device and the window size for the data acquisition.

Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2024 Michele Romani
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import re
import bluetooth
import logging
from brainflow import BoardIds

log_labels = {0: 'NOTSET', 1: 'DEBUG', 2: 'INFO', 3: 'WARNING', 4: 'ERROR', 5: 'CRITICAL'}


def retrieve_eeg_devices():
    saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
    unicorn_devices = list(filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices))
    logging.info(f"Found Unicorns: {unicorn_devices} ")
    enophone_devices = list(filter(lambda x: re.search(r'enophone', x[1]), saved_devices))
    logging.info(f"Found Enophones: {enophone_devices} ")
    synthetic_devices = [('00:00:00:00:00:00', 'Synthetic Board', '0000')]
    ant_neuro_devices = [('ANT_NEURO_225', 'ANT Neuro 225', '0000'), ('ANT_NEURO_411', 'ANT Neuro 411', '0000')]
    all_devices = synthetic_devices + unicorn_devices + enophone_devices + ant_neuro_devices
    return all_devices


def retrieve_board_id(device_name):
    if re.search(r'UN-\d{4}.\d{2}.\d{2}', device_name):
        return BoardIds.UNICORN_BOARD
    elif re.search(r'(?i)enophone', device_name):
        return BoardIds.ENOPHONE_BOARD
    elif re.search(r'(?i)ANT.NEURO.225', device_name):
        return BoardIds.ANT_NEURO_EE_225_BOARD
    elif re.search(r'(?i)ANT.NEURO.411', device_name):
        return BoardIds.ANT_NEURO_EE_411_BOARD
    else:
        return BoardIds.SYNTHETIC_BOARD


class SetupDialog(QtWidgets.QDialog):
    def __init__(self, devices, parent=None):
        super(SetupDialog, self).__init__(parent)

        self.setWindowTitle('Connect EEG')

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)

        # Create dropdown for device selection
        self.device_combo = QtWidgets.QComboBox(self)
        self.device_combo.addItems([device[1] for device in devices])
        layout.addWidget(QtWidgets.QLabel('Select device'))
        layout.addWidget(self.device_combo)

        # Create slider for window size
        self.window_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.window_size_slider.setMinimum(1)
        self.window_size_slider.setMaximum(20)
        self.window_size_slider.setValue(1)
        self.window_size_slider.valueChanged.connect(self.update_window_size_label)
        self.window_size_label = QtWidgets.QLabel(f'Window size: {self.window_size_slider.value()} seconds', self)

        layout.addWidget(self.window_size_label)
        layout.addWidget(self.window_size_slider)

        # Create slider for logging level
        self.logging_level_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.logging_level_slider.setMinimum(0)
        self.logging_level_slider.setMaximum(5)
        self.logging_level_slider.setValue(2)
        self.logging_level_slider.valueChanged.connect(self.update_logging_level_label)
        self.logging_level_label = QtWidgets.QLabel(f'Logging level: {log_labels[self.logging_level_slider.value()]} ',
                                                    self)

        layout.addWidget(self.logging_level_label)
        layout.addWidget(self.logging_level_slider)

        # Add OK and Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
                                                     self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def update_window_size_label(self, value):
        self.window_size_label.setText(f'Window size: {value} seconds')

    def update_logging_level_label(self, value):
        self.logging_level_label.setText(f'Logging level: {log_labels[value]} ')

    def get_data(self):
        return (
            self.device_combo.currentText(),
            self.window_size_slider.value(),
            self.logging_level_slider.value(),
        )
