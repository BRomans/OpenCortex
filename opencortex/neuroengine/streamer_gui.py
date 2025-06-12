"""
This class creates a GUI to plot and handle LSL events.
Filters can be applied to the data and the user can send triggers and save the data to a .CSV file.

Author: Michele Romani
Email: michele.romani.zaltieri@gmail.com
Copyright 2024 Michele Romani
"""

import threading
import time
import numpy as np
import logging
import pyqtgraph as pg
import os
import yaml
from PyQt5 import QtWidgets, QtCore
from opencortex.neuroengine.classifier import Classifier
from opencortex.neuroengine.core.stream_engine import StreamEngine
from opencortex.neuroengine.flux.base import Parallel
from opencortex.neuroengine.flux.band_power import BandPowerExtractor
from opencortex.neuroengine.flux.quality_estimator import QualityEstimator
from opencortex.neuroengine.gui.frequency_band_widget import FrequencyBandPanel
from opencortex.neuroengine.gui.gui_adapter import GUIAdapter
from opencortex.neuroengine.network.lsl_stream import LSLStreamThread, start_lsl_eeg_stream, start_lsl_power_bands_stream, \
    start_lsl_inference_stream, start_lsl_quality_stream, push_lsl_raw_eeg, push_lsl_band_powers, push_lsl_inference, \
    push_lsl_quality
from pyqtgraph import ScatterPlotItem, mkBrush
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from concurrent.futures import ThreadPoolExecutor

from opencortex.neuroengine.network.osc_stream import OscStreamThread
from opencortex.processing.preprocessing import extract_band_powers
from opencortex.processing.proc_helper import freq_bands
from opencortex.utils.layouts import layouts

colors = ["blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray",
          "cyan", "magenta", "lime", "teal", "lavender", "turquoise", "maroon", "olive",
          "blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray",
          "cyan", "magenta", "lime", "teal", "lavender", "turquoise", "maroon", "olive"]


def write_header(file, board_id):
    for column in layouts[board_id]["header"]:
        file.write(str(column) + '\t')
    file.write('\n')


class StreamerGUI:

    def __init__(self, board, params, window_size=1, config_file='default_config.yaml'):
        # Load configuration from file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        time.sleep(window_size)  # Wait for the board to be ready

        # 1. Create and start StreamEngine service
        self.stream_engine = StreamEngine(board, config, window_size)
        self.stream_engine.start()  # Start the service

        # 2. Create GUI adapter
        self.gui_adapter = GUIAdapter(self.stream_engine)

        # 3. Connect adapter signals to GUI methods
        self.gui_adapter.data_updated.connect(self.on_data_updated)
        self.gui_adapter.quality_updated.connect(self.on_quality_updated)
        self.gui_adapter.prediction_ready.connect(self.on_prediction_ready)
        self.gui_adapter.classifier_ready.connect(self.on_classifier_ready)
        self.gui_adapter.error_occurred.connect(self.on_error)
        self.gui_adapter.inference_mode_changed.connect(self.on_inference_mode_changed)


        self.window_size = window_size
        # Apply configuration
        self.plot = config.get('plot', True)
        self.save_data = config.get('save_data', True)
        self.model = config.get('model', 'LDA')
        self.proba = config.get('proba', False)
        self.group_predictions = config.get('group_predictions', False)
        self.nclasses = config.get('nclasses', 3)
        self.flash_time = config.get('flash_time', 250)
        self.epoch_length_ms = config.get('epoch_length_ms', 1000)
        self.baseline_ms = config.get('baseline_ms', 100)
        self.quality_thresholds = config.get('quality_thresholds', [(-100, -50, 'yellow', 0.5), (-50, 50, 'green', 1.0),
                                                                    (50, 100, 'yellow', 0.5)])
        self.update_buffer_speed_ms = config.get('update_buffer_speed_ms', 1000)
        self.update_plot_speed_ms = config.get('update_plot_speed_ms', 1000 / self.window_size)


        self.is_streaming = True
        self.inference_mode = False
        self.first_prediction = True
        self.lsl_state = True
        self.osc_state = False
        self.osc_thread = None
        self.params = params
        self.initial_ts = time.time()
        logging.info("Searching for devices...")
        self.board = board
        self.board_id = self.board.get_board_id()
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        logging.info(f"EEG channels: {BoardShim.get_eeg_names(self.board_id)}")
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.chunk_counter = 0
        self.num_points = self.window_size * self.sampling_rate
        self.filtered_eeg = np.zeros((len(self.eeg_channels) + 1, self.num_points))
        logging.info(f"Connected to {self.board.get_device_name(self.board.get_board_id())}")

        # Initialize the classifier in a new thread
        self.classifier = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        if self.model is not None:
            self.over_sample = config.get('oversample', True)
            self.classifier_thread = threading.Thread(target=self.init_classifier)
            self.classifier_thread.start()

        self.app = QtWidgets.QApplication([])

        # Calculate time interval for prediction
        self.off_time = (self.flash_time * (self.nclasses - 1))
        logging.debug(f"Off time: {self.off_time} ms")
        self.prediction_interval = int(
            2 * self.flash_time + self.off_time)  # * 3  # we take double the time, so we can loop on it
        logging.info(f"Prediction interval: {self.prediction_interval} ms")
        self.epoch_data_points = int(self.epoch_length_ms * self.sampling_rate / 1000)

        self.inference_ms = self.baseline_ms + (self.flash_time * self.nclasses) + self.epoch_length_ms

        self.prediction_datapoints = int(self.inference_ms * self.sampling_rate / 1000)
        self.slicing_trigger = (self.epoch_length_ms + self.baseline_ms) // self.flash_time
        if self.slicing_trigger > self.nclasses:
            self.slicing_trigger = self.nclasses
        logging.debug(f"Prediction interval in datapoints: {self.prediction_datapoints}")

        self.pipeline = Parallel(
            band_power=BandPowerExtractor(fs=self.sampling_rate, ch_names=self.eeg_channels),
            quality=QualityEstimator(quality_thresholds=self.quality_thresholds)
        )

        # Connect to the LSL stream threads
        self.prediction_timer = QtCore.QTimer()
        self.prediction_timer.timeout.connect(self.get_prediction_data)
        self.lsl_thread = LSLStreamThread()
        self.lsl_thread.new_sample.connect(self.write_trigger)
        self.lsl_thread.set_train_start.connect(self.set_train_start)
        self.lsl_thread.start_train.connect(self.train_classifier)
        self.lsl_thread.start_predicting.connect(self.set_inference_mode)
        self.lsl_thread.stop_predicting.connect(self.set_inference_mode)
        self.lsl_thread.start()



        self.win = pg.GraphicsLayoutWidget(title='OpenCortex Streamer', size=(1920, 1080))
        self.win.setWindowTitle('OpenCortex Streamer')
        self.win.show()
        lsl_panel = self.create_lsl_panel()
        processing_panel = self.create_parameters_panel()
        osc_panel = self.create_osc_panel()
        plot = self.init_plot()

        self.freq_band_panel = FrequencyBandPanel()
        self.freq_band_panel.bandsChanged.connect(self.on_frequency_bands_changed)

        side_panel_widget = QtWidgets.QWidget()
        side_panel_layout = QtWidgets.QVBoxLayout()
        side_panel_layout.addWidget(processing_panel)
        side_panel_widget.setLayout(side_panel_layout)

        bottom_panel_widget = QtWidgets.QWidget()
        bottom_panel_layout = QtWidgets.QHBoxLayout()
        bottom_panel_layout.addWidget(lsl_panel)
        bottom_panel_layout.addWidget(osc_panel)
        bottom_panel_layout.addWidget(self.freq_band_panel)
        bottom_panel_widget.setLayout(bottom_panel_layout)


        # Create a layout for the main window
        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(plot, 0, 0)
        self.main_layout.addWidget(bottom_panel_widget, 1, 0, alignment=QtCore.Qt.AlignLeft)
        self.main_layout.addWidget(side_panel_widget,0,1, alignment=QtCore.Qt.AlignCenter)

        # Set the main layout for the window
        self.win.setLayout(self.main_layout)

        # Start the PyQt event loop to fetch the raw data
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data_buffer)
        self.timer.start(self.update_buffer_speed_ms)

        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(self.update_plot_speed_ms)

        # Initialize LSL streams
        self.eeg_outlet = start_lsl_eeg_stream(channels=BoardShim.get_eeg_names(self.board_id), fs=self.sampling_rate,
                                               source_id=self.board.get_device_name(self.board_id))
        self.inference_outlet = start_lsl_inference_stream(channels=1,
                                                           fs=self.sampling_rate,
                                                           source_id=self.board.get_device_name(self.board_id))
        self.band_powers_outlet = start_lsl_power_bands_stream(channels=BoardShim.get_eeg_names(self.board_id),
                                                               fs=self.sampling_rate,
                                                               source_id=self.board.get_device_name(self.board_id))
        self.quality_outlet = start_lsl_quality_stream(channels=BoardShim.get_eeg_names(self.board_id),
                                                       fs=self.sampling_rate,
                                                       source_id=self.board.get_device_name(self.board_id))

        self.app.exec_()

    def create_lsl_panel(self):
        """Create a modern panel for LSL controls"""
        lsl_panel = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Main LSL group
        lsl_group = QtWidgets.QGroupBox()
        lsl_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #3a3a3a;
            }
        """)
        lsl_layout = QtWidgets.QVBoxLayout()

        # Section title
        lsl_controls_label = QtWidgets.QLabel("LSL Stream Controls")
        lsl_controls_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        lsl_controls_label.setAlignment(QtCore.Qt.AlignCenter)
        lsl_layout.addWidget(lsl_controls_label)

        # Status indicator
        self.lsl_status = QtWidgets.QLabel("●  LSL: Stopped")
        self.lsl_status.setStyleSheet("""
            QLabel {
                color: #f44336;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                background-color: #2a2a2a;
                border-radius: 4px;
            }
        """)
        self.lsl_status.setAlignment(QtCore.Qt.AlignCenter)
        lsl_layout.addWidget(self.lsl_status)

        # Trigger section
        trigger_container = QtWidgets.QWidget()
        trigger_layout = QtWidgets.QVBoxLayout()
        trigger_layout.setSpacing(8)

        trigger_label = QtWidgets.QLabel("Trigger Control")
        trigger_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        trigger_layout.addWidget(trigger_label)

        # Trigger input with label
        trigger_input_layout = QtWidgets.QVBoxLayout()
        trigger_input_layout.setSpacing(3)

        input_label = QtWidgets.QLabel("Trigger Value")
        input_label.setStyleSheet("color: #bbb; font-size: 10px;")

        self.input_box = QtWidgets.QLineEdit()
        self.input_box.setPlaceholderText('Enter value (1-255)')
        self.input_box.setText('1')
        self.input_box.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px 12px;
                color: white;
                font-size: 12px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
                background-color: #525252;
            }
            QLineEdit:hover {
                border-color: #777;
            }
        """)

        trigger_input_layout.addWidget(input_label)
        trigger_input_layout.addWidget(self.input_box)
        trigger_layout.addLayout(trigger_input_layout)

        # Send trigger button
        self.trigger_button = QtWidgets.QPushButton('📡 Send Trigger')
        self.trigger_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        self.trigger_button.clicked.connect(self.send_trigger_safely)
        trigger_layout.addWidget(self.trigger_button)

        trigger_container.setLayout(trigger_layout)
        lsl_layout.addWidget(trigger_container)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setStyleSheet("color: #666; margin: 5px 0;")
        lsl_layout.addWidget(separator)

        # Options section
        options_layout = QtWidgets.QVBoxLayout()
        options_layout.setSpacing(8)

        options_label = QtWidgets.QLabel("Stream Options")
        options_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        options_layout.addWidget(options_label)

        # Chunk data checkbox
        self.lsl_chunk_checkbox = QtWidgets.QCheckBox('Enable Data Chunking')
        self.lsl_chunk_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3;
                border: 2px solid #1976D2;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 3px;
            }
        """)
        self.lsl_chunk_checkbox.setChecked(True)
        options_layout.addWidget(self.lsl_chunk_checkbox)

        lsl_layout.addLayout(options_layout)

        # Start/Stop LSL button
        self.lsl_button = QtWidgets.QPushButton('🚀 Start LSL Stream')
        self.lsl_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.lsl_button.clicked.connect(self.toggle_lsl)
        lsl_layout.addWidget(self.lsl_button)

        lsl_group.setLayout(lsl_layout)
        main_layout.addWidget(lsl_group)
        main_layout.addStretch()

        # Set panel properties
        lsl_panel.setLayout(main_layout)
        lsl_panel.setMinimumWidth(320)
        lsl_panel.setMaximumWidth(380)
        lsl_panel.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
        """)

        return lsl_panel

    def create_osc_panel(self):
        """Create a modern panel for OSC controls"""
        osc_panel = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Main OSC group
        osc_group = QtWidgets.QGroupBox()
        osc_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #3a3a3a;
            }
        """)
        osc_layout = QtWidgets.QVBoxLayout()

        # Section title
        osc_controls_label = QtWidgets.QLabel("OSC Communication")
        osc_controls_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        osc_controls_label.setAlignment(QtCore.Qt.AlignCenter)
        osc_layout.addWidget(osc_controls_label)

        # Status indicator
        self.osc_status = QtWidgets.QLabel("●  OSC: Disconnected")
        self.osc_status.setStyleSheet("""
            QLabel {
                color: #f44336;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                background-color: #2a2a2a;
                border-radius: 4px;
            }
        """)
        self.osc_status.setAlignment(QtCore.Qt.AlignCenter)
        osc_layout.addWidget(self.osc_status)

        # Connection settings section
        connection_container = QtWidgets.QWidget()
        connection_layout = QtWidgets.QVBoxLayout()
        connection_layout.setSpacing(10)

        connection_label = QtWidgets.QLabel("Connection Settings")
        connection_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        connection_layout.addWidget(connection_label)

        # OSC Address input
        address_input_layout = QtWidgets.QVBoxLayout()
        address_input_layout.setSpacing(3)

        address_label = QtWidgets.QLabel("Target Address")
        address_label.setStyleSheet("color: #bbb; font-size: 10px;")

        self.osc_address_input = QtWidgets.QLineEdit()
        self.osc_address_input.setText("127.0.0.1")
        self.osc_address_input.setPlaceholderText("IP Address")
        self.osc_address_input.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px 12px;
                color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #FF9800;
                background-color: #525252;
            }
            QLineEdit:hover {
                border-color: #777;
            }
        """)

        address_input_layout.addWidget(address_label)
        address_input_layout.addWidget(self.osc_address_input)
        connection_layout.addLayout(address_input_layout)

        # Port settings in a grid layout
        ports_container = QtWidgets.QWidget()
        ports_layout = QtWidgets.QHBoxLayout()
        ports_layout.setSpacing(10)

        # Listen port
        listen_layout = QtWidgets.QVBoxLayout()
        listen_layout.setSpacing(3)

        port_input_label = QtWidgets.QLabel("Listen Port")
        port_input_label.setStyleSheet("color: #bbb; font-size: 10px;")

        self.osc_port_input = QtWidgets.QLineEdit()
        self.osc_port_input.setText("8000")
        self.osc_port_input.setPlaceholderText("8000")
        self.osc_port_input.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px 8px;
                color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)

        listen_layout.addWidget(port_input_label)
        listen_layout.addWidget(self.osc_port_input)

        # Send port
        send_layout = QtWidgets.QVBoxLayout()
        send_layout.setSpacing(3)

        port_output_label = QtWidgets.QLabel("Send Port")
        port_output_label.setStyleSheet("color: #bbb; font-size: 10px;")

        self.osc_port_output = QtWidgets.QLineEdit()
        self.osc_port_output.setText("9000")
        self.osc_port_output.setPlaceholderText("9000")
        self.osc_port_output.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px 8px;
                color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #FF5722;
            }
        """)

        send_layout.addWidget(port_output_label)
        send_layout.addWidget(self.osc_port_output)

        ports_layout.addLayout(listen_layout)
        ports_layout.addLayout(send_layout)
        ports_container.setLayout(ports_layout)
        connection_layout.addWidget(ports_container)

        connection_container.setLayout(connection_layout)
        osc_layout.addWidget(connection_container)

        # Connection info display
        self.osc_info_label = QtWidgets.QLabel("Ready to connect")
        self.osc_info_label.setStyleSheet("""
            QLabel {
                color: #bbb;
                font-size: 10px;
                font-style: italic;
                padding: 5px;
                background-color: #2a2a2a;
                border-radius: 3px;
                margin: 5px 0;
            }
        """)
        self.osc_info_label.setAlignment(QtCore.Qt.AlignCenter)
        osc_layout.addWidget(self.osc_info_label)

        # Start/Stop OSC button
        self.osc_button = QtWidgets.QPushButton('🌐 Start OSC Communication')
        self.osc_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
        """)
        self.osc_button.clicked.connect(self.toggle_osc)
        osc_layout.addWidget(self.osc_button)

        osc_group.setLayout(osc_layout)
        main_layout.addWidget(osc_group)
        main_layout.addStretch()

        # Set panel properties
        osc_panel.setLayout(main_layout)
        osc_panel.setMinimumWidth(320)
        osc_panel.setMaximumWidth(380)
        osc_panel.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
        """)

        return osc_panel

    # Helper methods for improved functionality
    def send_trigger_safely(self):
        """Send trigger with validation and feedback"""
        try:
            trigger_value = int(self.input_box.text())
            if 1 <= trigger_value <= 255:
                self.write_trigger(trigger_value)
                # Visual feedback
                original_style = self.trigger_button.styleSheet()
                self.trigger_button.setStyleSheet(original_style.replace('#4CAF50', '#66BB6A'))
                QtCore.QTimer.singleShot(200, lambda: self.trigger_button.setStyleSheet(original_style))
            else:
                self.show_trigger_error("Trigger value must be between 1-255")
        except ValueError:
            self.show_trigger_error("Please enter a valid number")

    def show_trigger_error(self, message):
        """Show trigger input error"""
        original_style = self.input_box.styleSheet()
        error_style = original_style.replace('#666', '#f44336').replace('#4CAF50', '#f44336')
        self.input_box.setStyleSheet(error_style)
        self.input_box.setPlaceholderText(message)
        QtCore.QTimer.singleShot(2000, lambda: [
            self.input_box.setStyleSheet(original_style),
            self.input_box.setPlaceholderText('Enter value (1-255)')
        ])

    def update_lsl_status(self, connected=False):
        """Update LSL status indicator"""
        if hasattr(self, 'lsl_status'):
            if connected:
                self.lsl_status.setText("●  LSL: Connected")
                self.lsl_status.setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 5px;
                        background-color: #2a2a2a;
                        border-radius: 4px;
                    }
                """)
                self.lsl_button.setText('🛑 Stop LSL Stream')
                self.trigger_button.setEnabled(True)
            else:
                self.lsl_status.setText("●  LSL: Stopped")
                self.lsl_status.setStyleSheet("""
                    QLabel {
                        color: #f44336;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 5px;
                        background-color: #2a2a2a;
                        border-radius: 4px;
                    }
                """)
                self.lsl_button.setText('🚀 Start LSL Stream')
                self.trigger_button.setEnabled(False)

    def update_osc_status(self, connected=False, address="", listen_port="", send_port=""):
        """Update OSC status indicator"""
        if hasattr(self, 'osc_status'):
            if connected:
                self.osc_status.setText("●  OSC: Connected")
                self.osc_status.setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 5px;
                        background-color: #2a2a2a;
                        border-radius: 4px;
                    }
                """)
                self.osc_button.setText('🛑 Stop OSC Communication')
                self.osc_info_label.setText(f"Connected: {address}:{listen_port} ⇄ {send_port}")

                # Disable inputs when connected
                self.osc_address_input.setEnabled(False)
                self.osc_port_input.setEnabled(False)
                self.osc_port_output.setEnabled(False)
            else:
                self.osc_status.setText("●  OSC: Disconnected")
                self.osc_status.setStyleSheet("""
                    QLabel {
                        color: #f44336;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 5px;
                        background-color: #2a2a2a;
                        border-radius: 4px;
                    }
                """)
                self.osc_button.setText('🌐 Start OSC Communication')
                self.osc_info_label.setText("Ready to connect")

                # Enable inputs when disconnected
                self.osc_address_input.setEnabled(True)
                self.osc_port_input.setEnabled(True)
                self.osc_port_output.setEnabled(True)

    # Update your existing toggle methods to use the new status updates
    def toggle_lsl(self):
        """Toggle LSL streaming on and off with visual feedback"""
        if self.lsl_state:
            self.lsl_thread.quit()
            self.update_lsl_status(connected=False)
            self.lsl_state = False
        else:
            self.lsl_thread.start()
            self.update_lsl_status(connected=True)
            self.lsl_state = True

    def toggle_osc(self):
        """Toggle OSC communication with visual feedback"""
        if not self.osc_state:
            listen_port = int(self.osc_port_input.text())
            send_port = int(self.osc_port_output.text())
            address = self.osc_address_input.text()

            self.osc_thread = OscStreamThread(listen_port=listen_port, send_port=send_port)
            self.osc_thread.message_received.connect(lambda addr, args: print(f"Received {addr}: {args}"))
            self.osc_thread.start()

            self.update_osc_status(connected=True, address=address,
                                   listen_port=str(listen_port), send_port=str(send_port))
            self.osc_state = True
        else:
            if self.osc_thread:
                self.osc_thread.stop()
                self.osc_thread = None
            self.update_osc_status(connected=False)
            self.osc_state = False

    def create_parameters_panel(self):
        """Create buttons to interact with the neuroengine - Modern styled version"""

        # Main container
        parameters = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ==================== EEG OPTIONS SECTION ====================
        eeg_group = QtWidgets.QGroupBox()
        eeg_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #3a3a3a;
            }
        """)
        eeg_layout = QtWidgets.QVBoxLayout()

        # Section title
        eeg_options_label = QtWidgets.QLabel("EEG Options")
        eeg_options_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        eeg_options_label.setAlignment(QtCore.Qt.AlignCenter)
        eeg_layout.addWidget(eeg_options_label)

        # Save data checkbox - styled
        self.save_data_checkbox = QtWidgets.QCheckBox('Save to CSV')
        self.save_data_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #45a049;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 3px;
            }
        """)
        self.save_data_checkbox.setChecked(self.save_data)
        eeg_layout.addWidget(self.save_data_checkbox)

        # Start/Stop plot button - modern style
        self.start_button = QtWidgets.QPushButton('Stop Plot')
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self.start_button.clicked.connect(lambda: self.toggle_plot())
        eeg_layout.addWidget(self.start_button)

        eeg_group.setLayout(eeg_layout)
        main_layout.addWidget(eeg_group)

        # ==================== FILTERS SECTION ====================
        filters_group = QtWidgets.QGroupBox()
        filters_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #3a3a3a;
            }
        """)
        filters_layout = QtWidgets.QVBoxLayout()

        # Section title
        filters_label = QtWidgets.QLabel("Signal Filters")
        filters_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        filters_label.setAlignment(QtCore.Qt.AlignCenter)
        filters_layout.addWidget(filters_label)

        # Bandpass filter section
        bandpass_container = QtWidgets.QWidget()
        bandpass_layout = QtWidgets.QVBoxLayout()
        bandpass_layout.setSpacing(8)

        self.bandpass_checkbox = QtWidgets.QCheckBox('Bandpass Filter (Hz)')
        self.bandpass_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3;
                border: 2px solid #1976D2;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 3px;
            }
        """)
        bandpass_layout.addWidget(self.bandpass_checkbox)

        # Bandpass frequency inputs
        bandpass_inputs_layout = QtWidgets.QHBoxLayout()
        bandpass_inputs_layout.setSpacing(10)

        # Low frequency input
        low_freq_layout = QtWidgets.QVBoxLayout()
        low_freq_layout.setSpacing(2)
        low_label = QtWidgets.QLabel("Low")
        low_label.setStyleSheet("color: #bbb; font-size: 10px;")
        low_label.setAlignment(QtCore.Qt.AlignCenter)

        self.bandpass_box_low = QtWidgets.QLineEdit()
        self.bandpass_box_low.setPlaceholderText('1.0')
        self.bandpass_box_low.setText('1')
        self.bandpass_box_low.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 6px;
                color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
        """)
        self.bandpass_box_low.setMaximumWidth(50)

        low_freq_layout.addWidget(low_label)
        low_freq_layout.addWidget(self.bandpass_box_low)

        # High frequency input
        high_freq_layout = QtWidgets.QVBoxLayout()
        high_freq_layout.setSpacing(2)
        high_label = QtWidgets.QLabel("High")
        high_label.setStyleSheet("color: #bbb; font-size: 10px;")
        high_label.setAlignment(QtCore.Qt.AlignCenter)

        self.bandpass_box_high = QtWidgets.QLineEdit()
        self.bandpass_box_high.setPlaceholderText('40.0')
        self.bandpass_box_high.setText('40')
        self.bandpass_box_high.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 6px;
                color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
        """)
        self.bandpass_box_high.setMaximumWidth(50)

        high_freq_layout.addWidget(high_label)
        high_freq_layout.addWidget(self.bandpass_box_high)

        bandpass_inputs_layout.addLayout(low_freq_layout)
        bandpass_inputs_layout.addWidget(QtWidgets.QLabel("-"))  # Separator
        bandpass_inputs_layout.addLayout(high_freq_layout)
        bandpass_inputs_layout.addStretch()

        bandpass_layout.addLayout(bandpass_inputs_layout)
        bandpass_container.setLayout(bandpass_layout)
        filters_layout.addWidget(bandpass_container)

        # Separator line
        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.HLine)
        separator1.setStyleSheet("color: #666;")
        filters_layout.addWidget(separator1)

        # Notch filter section
        notch_container = QtWidgets.QWidget()
        notch_layout = QtWidgets.QVBoxLayout()
        notch_layout.setSpacing(8)

        self.notch_checkbox = QtWidgets.QCheckBox('Notch Filter (Hz)')
        self.notch_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:checked {
                background-color: #FF9800;
                border: 2px solid #F57C00;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #555;
                border: 2px solid #777;
                border-radius: 3px;
            }
        """)
        notch_layout.addWidget(self.notch_checkbox)

        # Notch frequency input
        notch_freq_layout = QtWidgets.QVBoxLayout()
        notch_freq_layout.setSpacing(2)
        notch_label = QtWidgets.QLabel("Frequencies (comma separated)")
        notch_label.setStyleSheet("color: #bbb; font-size: 10px;")

        self.notch_box = QtWidgets.QLineEdit()
        self.notch_box.setPlaceholderText('50, 60')
        self.notch_box.setText('50, 60')
        self.notch_box.setStyleSheet("""
            QLineEdit {
                background-color: #4a4a4a;
                border: 2px solid #666;
                border-radius: 4px;
                padding: 6px;
                color: white;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #FF9800;
            }
        """)

        notch_freq_layout.addWidget(notch_label)
        notch_freq_layout.addWidget(self.notch_box)
        notch_layout.addLayout(notch_freq_layout)

        notch_container.setLayout(notch_layout)
        filters_layout.addWidget(notch_container)

        filters_group.setLayout(filters_layout)
        main_layout.addWidget(filters_group)

        # ==================== CLASSIFIER SECTION ====================
        classifier_group = QtWidgets.QGroupBox()
        classifier_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
                background-color: #3a3a3a;
            }
        """)
        classifier_layout = QtWidgets.QVBoxLayout()

        # Section title
        model_label = QtWidgets.QLabel("Classifier Analysis")
        model_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        model_label.setAlignment(QtCore.Qt.AlignCenter)
        classifier_layout.addWidget(model_label)

        # Classifier buttons layout
        classifier_buttons_layout = QtWidgets.QHBoxLayout()
        classifier_buttons_layout.setSpacing(10)

        # ROC button
        self.roc_button = QtWidgets.QPushButton('Plot ROC')
        self.roc_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:pressed {
                background-color: #4A148C;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        self.roc_button.clicked.connect(lambda: self.classifier.plot_roc_curve())
        classifier_buttons_layout.addWidget(self.roc_button)

        # Confusion Matrix button
        self.confusion_button = QtWidgets.QPushButton('Plot CM')
        self.confusion_button.setStyleSheet("""
            QPushButton {
                background-color: #673AB7;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #512DA8;
            }
            QPushButton:pressed {
                background-color: #311B92;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        self.confusion_button.clicked.connect(lambda: self.classifier.plot_confusion_matrix())
        classifier_buttons_layout.addWidget(self.confusion_button)

        classifier_layout.addLayout(classifier_buttons_layout)

        # Add classifier status indicator
        self.classifier_status = QtWidgets.QLabel("Classifier: Not Ready")
        self.classifier_status.setStyleSheet("""
            QLabel {
                color: #ff9800;
                font-size: 10px;
                font-style: italic;
                padding: 5px;
                background-color: #2a2a2a;
                border-radius: 3px;
                margin-top: 5px;
            }
        """)
        self.classifier_status.setAlignment(QtCore.Qt.AlignCenter)
        classifier_layout.addWidget(self.classifier_status)

        classifier_group.setLayout(classifier_layout)
        main_layout.addWidget(classifier_group)

        # ==================== SPACER AND FINAL SETUP ====================
        main_layout.addStretch()  # Push everything to the top

        # Set the main layout
        parameters.setLayout(main_layout)
        parameters.setMinimumWidth(320)
        parameters.setMaximumWidth(380)
        parameters.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)

        # Connect filter checkboxes to enable/disable inputs
        self.bandpass_checkbox.stateChanged.connect(self.on_bandpass_toggled)
        self.notch_checkbox.stateChanged.connect(self.on_notch_toggled)

        # Initial state
        self.on_bandpass_toggled(self.bandpass_checkbox.checkState())
        self.on_notch_toggled(self.notch_checkbox.checkState())

        return parameters

    def on_bandpass_toggled(self, state):
        """Enable/disable bandpass filter inputs based on checkbox"""
        enabled = state == QtCore.Qt.Checked
        self.bandpass_box_low.setEnabled(enabled)
        self.bandpass_box_high.setEnabled(enabled)

        # Update visual state
        if enabled:
            self.bandpass_box_low.setStyleSheet(
                self.bandpass_box_low.styleSheet().replace('color: #666', 'color: white'))
            self.bandpass_box_high.setStyleSheet(
                self.bandpass_box_high.styleSheet().replace('color: #666', 'color: white'))
        else:
            # Make inputs look disabled
            disabled_style = """
                QLineEdit {
                    background-color: #333;
                    border: 2px solid #555;
                    border-radius: 4px;
                    padding: 6px;
                    color: #666;
                    font-size: 11px;
                }
            """
            self.bandpass_box_low.setStyleSheet(disabled_style)
            self.bandpass_box_high.setStyleSheet(disabled_style)

    def on_notch_toggled(self, state):
        """Enable/disable notch filter inputs based on checkbox"""
        enabled = state == QtCore.Qt.Checked
        self.notch_box.setEnabled(enabled)

        # Update visual state
        if enabled:
            self.notch_box.setStyleSheet("""
                QLineEdit {
                    background-color: #4a4a4a;
                    border: 2px solid #666;
                    border-radius: 4px;
                    padding: 6px;
                    color: white;
                    font-size: 11px;
                }
                QLineEdit:focus {
                    border-color: #FF9800;
                }
            """)
        else:
            self.notch_box.setStyleSheet("""
                QLineEdit {
                    background-color: #333;
                    border: 2px solid #555;
                    border-radius: 4px;
                    padding: 6px;
                    color: #666;
                    font-size: 11px;
                }
            """)

    def update_classifier_status(self, status="ready"):
        """Update the classifier status indicator"""
        if hasattr(self, 'classifier_status'):
            if status == "ready":
                self.classifier_status.setText("Classifier: Ready")
                self.classifier_status.setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        font-size: 10px;
                        font-style: italic;
                        padding: 5px;
                        background-color: #2a2a2a;
                        border-radius: 3px;
                        margin-top: 5px;
                    }
                """)
                self.roc_button.setEnabled(True)
                self.confusion_button.setEnabled(True)
            elif status == "training":
                self.classifier_status.setText("Classifier: Training...")
                self.classifier_status.setStyleSheet("""
                    QLabel {
                        color: #2196F3;
                        font-size: 10px;
                        font-style: italic;
                        padding: 5px;
                        background-color: #2a2a2a;
                        border-radius: 3px;
                        margin-top: 5px;
                    }
                """)
                self.roc_button.setEnabled(False)
                self.confusion_button.setEnabled(False)
            else:  # not ready
                self.classifier_status.setText("Classifier: Not Ready")
                self.classifier_status.setStyleSheet("""
                    QLabel {
                        color: #ff9800;
                        font-size: 10px;
                        font-style: italic;
                        padding: 5px;
                        background-color: #2a2a2a;
                        border-radius: 3px;
                        margin-top: 5px;
                    }
                """)
                self.roc_button.setEnabled(False)
                self.confusion_button.setEnabled(False)

    def create_frequency_band_panel(self):
        # Define the constraints for each band
        band_constraints = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 50)
        }

        self.freq_band_panel = FrequencyBandPanel(band_constraints)
        self.freq_band_panel.bandsChanged.connect(self.on_frequency_bands_changed)

        return self.freq_band_panel

    def set_inference_mode(self):
        """Set the BCI running status"""
        self.inference_mode = not self.inference_mode
        self.classifier.set_inference_mode(self.inference_mode)

    def update_data_buffer(self):
        """ Update the out stream buffer with new data"""
        if self.window_size == 0:
            raise ValueError("Window size cannot be zero")
        try:
            data = self.board.get_current_board_data(num_samples=self.num_points)
            self.filter_data_buffer(data)
            start_eeg = layouts[self.board_id]["eeg_start"]
            end_eeg = layouts[self.board_id]["eeg_end"]
            eeg = data[start_eeg:end_eeg]
            self.chunk_counter += 1

            for count, channel in enumerate(self.eeg_channels):
                ch_data = eeg[count]
                self.filtered_eeg[count] = ch_data

            trigger = data[-1]
            ts_channel = self.board.get_timestamp_channel(self.board_id)
            ts = data[ts_channel]
            self.filtered_eeg[-1] = trigger
        except Exception as e:
            logging.error(f"Error updating data buffer: {e}")
            return

        try:

            outputs = self.pipeline(self.filtered_eeg[0:len(self.eeg_channels)])
            band_powers = outputs["band_power"]
            quality_scores = outputs["quality"]

            push_lsl_band_powers(self.band_powers_outlet, band_powers.to_numpy(), ts)
            push_lsl_quality(self.quality_outlet, quality_scores)
            push_lsl_raw_eeg(self.eeg_outlet, self.filtered_eeg, start_eeg, end_eeg, self.chunk_counter, ts,
                             self.lsl_chunk_checkbox.isChecked())
            # Send a test message
            if self.osc_thread: self.osc_thread.send_message(self.osc_address_input.text(), band_powers.to_numpy().tolist())
        except Exception as e:
            logging.error(f"Error pushing data to LSL: {e}")
        self.app.processEvents()

    def init_plot(self):
        """Initialize the timeseries plot for the EEG channels and trigger channel."""

        # Initialize a single plot for all EEG channels including the trigger
        self.eeg_plot = pg.PlotWidget()  # Use PlotWidget to create a plot that can be added to a layout

        # Configure the plot settings
        self.eeg_plot.showAxis('left', False)  # Hide the Y-axis labels
        self.eeg_plot.setMenuEnabled('left', True)
        self.eeg_plot.showAxis('bottom', True)
        self.eeg_plot.setMenuEnabled('bottom', True)
        self.eeg_plot.showGrid(x=True, y=True)
        self.eeg_plot.setLabel('bottom', text='Time (s)')
        self.eeg_plot.getAxis('bottom').setTicks([[(i, str(i / self.sampling_rate)) for i in
                                                   range(0, self.num_points, int(self.sampling_rate / 2))] + [
                                                      (self.num_points, str(self.num_points / self.sampling_rate))]])

        self.eeg_plot.setTitle('EEG Channels with Trigger')

        # Set a smaller vertical offset to fit within the reduced height
        self.offset_amplitude = 200  # Adjusted for smaller plot height
        self.trigger_offset = -self.offset_amplitude  # Offset for the trigger channel

        # Initialize the curves and quality indicators for each channel
        self.curves = []
        self.quality_indicators = []

        for i, channel in enumerate(self.eeg_channels):
            # Plot each channel with a different color
            curve = self.eeg_plot.plot(pen=colors[i])
            self.curves.append(curve)

            # Create and add quality indicator
            scatter = ScatterPlotItem(size=20, brush=mkBrush('green'))
            # position the item according to the offset for each channel
            scatter.setPos(-1, i * self.offset_amplitude)
            self.eeg_plot.addItem(scatter)
            self.quality_indicators.append(scatter)

            # Add labels for each channel
            text_item = pg.TextItem(text=BoardShim.get_eeg_names(self.board_id)[i], anchor=(1, 0.5))
            text_item.setPos(-10, i * self.offset_amplitude)  # Position label next to the corresponding channel
            self.eeg_plot.addItem(text_item)

            # Add a small indicator for the uV range next to each channel
            uv_indicator = pg.TextItem(text=f"±{int(self.offset_amplitude / 2)} uV", anchor=(0, 1))
            uv_indicator.setPos(-10, i * self.offset_amplitude)  # Position the indicator on the right side
            #self.eeg_plot.addItem(uv_indicator)

        # Add the trigger curve at the bottom
        trigger_curve = self.eeg_plot.plot(pen='red')
        self.curves.append(trigger_curve)

        # Add label for the trigger channel
        trigger_label = pg.TextItem(text="Trigger", anchor=(1, 0.5))
        trigger_label.setPos(-10, self.trigger_offset)  # Position label next to the trigger channel
        self.eeg_plot.addItem(trigger_label)
        return self.eeg_plot

    def update_plot(self):
        """Update the plot with new data."""
        if self.is_streaming:
            filtered_eeg = np.zeros((len(self.eeg_channels) + 1, self.num_points))
            data = self.board.get_current_board_data(num_samples=self.num_points)
            if self.window_size == 0:
                raise ValueError("Window size cannot be zero")
            self.filter_data_buffer(data)
            start_eeg = layouts[self.board_id]["eeg_start"]
            end_eeg = layouts[self.board_id]["eeg_end"]
            eeg = data[start_eeg:end_eeg]

            for count, channel in enumerate(self.eeg_channels):
                ch_data = eeg[count]
                filtered_eeg[count] = ch_data

                if self.plot:
                    # Apply the offset for display purposes only
                    ch_data_offset = ch_data + count * self.offset_amplitude
                    self.curves[count].setData(ch_data_offset)

            # Plot the trigger channel, scaled and offset appropriately
            trigger = data[-1]
            if self.plot:
                # Rescale trigger to fit the display range and apply the offset
                trigger_rescaled = (trigger * (self.offset_amplitude / 5.0) + self.trigger_offset)
                self.curves[-1].setData(trigger_rescaled.tolist())

            # Adjust the Y range to fit all channels with their offsets and the trigger
            min_display = self.trigger_offset - self.offset_amplitude
            max_display = (len(self.eeg_channels)) * self.offset_amplitude
            self.eeg_plot.setYRange(min_display, max_display)

            self.update_quality_indicators(filtered_eeg, push=True)
            self.app.processEvents()

    def export_file(self, filename=None, folder='export', format='csv'):
        """
        Export the data to a file
        :param filename: str, name of the file
        :param folder: str, name of the folder
        :param format: str, format of the file
        """
        # Compose the file name using the board name and the current time
        try:
            if filename is None:
                filename = f"{self.board.get_device_name(self.board_id)}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            path = os.path.join(folder, filename + '.' + format)
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(path, 'w') as self.file:
                write_header(self.file, self.board_id)
                data = self.board.get_board_data()
                if format == 'csv':
                    DataFilter.write_file(data, path, 'a')
        except Exception as e:
            logging.error(f"Error exporting file: {e}")

    def write_trigger(self, trigger=1, timestamp=0):
        """
        Insert a trigger into the data stream
        :param trigger: int, trigger value
        :param timestamp: float, timestamp value
        """
        self.gui_adapter.send_trigger(trigger)


    def init_classifier(self):
        """ Initialize the classifier """
        self.classifier = Classifier(model=self.model, board_id=self.board_id)

    def set_train_start(self):
        """" Set the start of the training"""
        self.start_training_time = time.time()

    def train_classifier(self):
        """ Train the classifier"""
        self.gui_adapter.train_classifier()

    def start_prediction_timer(self):
        """Start the prediction timer."""
        self.prediction_timer.start(self.prediction_interval)

    def stop_prediction_timer(self):
        """Stop the prediction timer."""
        self.prediction_timer.stop()

    def get_prediction_data(self):

        inference_sample = self.board.get_current_board_data(self.prediction_datapoints)
        self.filter_data_buffer(inference_sample)
        logging.debug(f"Inference length: {self.inference_ms} Prediction datapoints {self.prediction_datapoints}")

    def _predict_class(self, data):
        """Internal method to predict the class of the data."""
        try:
            output = self.classifier.predict(data, proba=self.proba, group=self.group_predictions)
            push_lsl_inference(self.inference_outlet, output)
            logging.info(f"Predicted class: {output}")
        except Exception as e:
            logging.error(f"Error predicting class: {e}")

    def predict_class(self):
        """Predict the class of the data."""
        try:
            inference_sample = self.board.get_current_board_data(self.prediction_datapoints)
            self.filter_data_buffer(inference_sample)
            self.executor.submit(self._predict_class, inference_sample)
        except Exception as e:
            logging.error(f"Error starting prediction task: {e}")

    def update_quality_indicators(self, sample, push=False):
        """ Update the quality indicators for each channel"""
        eeg_start = layouts[self.board_id]["eeg_start"]
        eeg_end = layouts[self.board_id]["eeg_end"]
        eeg = sample[eeg_start:eeg_end]
        amplitudes = []
        q_colors = []
        q_scores = []
        for i in range(len(eeg)):
            amplitude_data = eeg[i]  # get the data for the i-th channel
            color, amplitude, q_score = self.get_channel_quality(amplitude_data)
            q_colors.append(color)
            amplitudes.append(np.round(amplitude, 2))
            q_scores.append(q_score)
            # Update the scatter plot item with the new color
            self.quality_indicators[i].setBrush(mkBrush(color))
            self.quality_indicators[i].setData([-1], [0])  # Position the circle at (0, 0)
        if push:
            #push_lsl_quality(self.quality_outlet, q_scores)
            pass
        logging.debug(f"Qualities: {q_scores} {q_colors}")

    def get_channel_quality(self, eeg, threshold=75):
        """ Get the quality of the EEG channel based on the amplitude"""
        amplitude = np.percentile(eeg, threshold)
        q_score = 0
        color = 'red'
        for low, high, color_name, score in self.quality_thresholds:
            if low <= amplitude <= high:
                color = color_name
                q_score = score
                break
        return color, amplitude, q_score

    def filter_data_buffer(self, data):
        start_eeg = layouts[self.board_id]["eeg_start"]
        end_eeg = layouts[self.board_id]["eeg_end"]
        eeg = data[start_eeg:end_eeg]
        for count, channel in enumerate(self.eeg_channels):
            ch_data = eeg[count]
            if self.bandpass_checkbox.isChecked():
                start_freq = float(self.bandpass_box_low.text()) if self.bandpass_box_low.text() != '' else 0
                stop_freq = float(self.bandpass_box_high.text()) if self.bandpass_box_high.text() != '' else 0
                self.apply_bandpass_filter(ch_data, start_freq, stop_freq)
            if self.notch_checkbox.isChecked():
                freqs = np.array(self.notch_box.text().split(','))
                self.apply_notch_filter(ch_data, freqs)

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

    def toggle_stream(self):
        """ Start or stop the streaming of data"""
        if self.is_streaming:
            self.board.stop_stream()
            self.is_streaming = False
        else:
            self.board.start_stream()
            self.is_streaming = True

    def toggle_plot(self):
        """ Start or stop the streaming of data"""
        if self.plot:
            self.start_button.setText('Start Plotting')
            self.plot = False
        else:
            self.start_button.setText('Stop Plotting')
            self.plot = True

    def quit(self):
        """ Quit the neuroengine, join the threads and stop the streaming"""
        if self.save_data_checkbox.isChecked():
            logging.info("Exporting data to file")
            self.export_file()
        self.lsl_thread.quit()
        self.classifier_thread.join()
        self.board.stop_stream()

    def on_data_updated(self, data):
        """Handle data updates from StreamEngine"""
        # For now, just log that we received data
        logging.debug("Received data update from StreamEngine")

    def on_quality_updated(self, quality_scores):
        """Handle quality updates from StreamEngine"""
        logging.debug(f"Received quality update: {quality_scores}")

    def on_prediction_ready(self, prediction):
        """Handle prediction results from StreamEngine"""
        logging.info(f"Received prediction: {prediction}")

    def on_classifier_ready(self):
        """Handle classifier initialization complete"""
        logging.info("StreamEngine classifier is ready")

    def on_error(self, error_msg):
        """Handle errors from StreamEngine"""
        logging.error(f"StreamEngine error: {error_msg}")

    def on_inference_mode_changed(self, mode):
        """Handle inference mode changes from StreamEngine"""
        logging.info(f"Inference mode changed: {mode}")

    def on_frequency_bands_changed(self, freq_bands):
        """Handle frequency band changes"""
        logging.info(f"Updated frequency bands: {freq_bands}")

        # Update your band power extractor
        if hasattr(self, 'pipeline'):
            self.pipeline.band_power.update_frequency_bands(freq_bands)

