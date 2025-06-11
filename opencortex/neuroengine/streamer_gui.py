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

        # ============ NEW SERVICE-BASED APPROACH ============

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

        time.sleep(self.window_size)  # Wait for the board to be ready
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
        panel = self.create_parameters_panel()
        osc_panel = self.create_osc_panel()
        plot = self.init_plot()

        side_panel_widget = QtWidgets.QWidget()
        side_panel_layout = QtWidgets.QVBoxLayout()
        side_panel_layout.addWidget(lsl_panel)
        side_panel_layout.addWidget(osc_panel)
        side_panel_layout.addWidget(panel)
        side_panel_widget.setLayout(side_panel_layout)

        # Create a layout for the main window
        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(plot, 0, 0)
        self.main_layout.addWidget(side_panel_widget,0, 1, alignment=QtCore.Qt.AlignCenter)

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
        """Create a panel for LSL controls"""
        lsl_panel = QtWidgets.QWidget()
        lsl_layout = QtWidgets.QVBoxLayout()

        # Create a button to start/stop LSL
        self.lsl_button = QtWidgets.QPushButton('Start LSL')
        self.lsl_button.setFixedWidth(100)
        self.lsl_button.clicked.connect(self.toggle_lsl)

        # Button to write trigger and input box to specify the trigger value
        self.input_box = QtWidgets.QLineEdit()
        self.input_box.setFixedWidth(100)  # Set a fixed width for the input box
        self.input_box.setPlaceholderText('Trigger value')
        self.input_box.setText('1')

        self.trigger_button = QtWidgets.QPushButton('Send Trigger')
        self.trigger_button.setFixedWidth(100)  # Set a fixed width for the button
        self.trigger_button.clicked.connect(lambda: self.write_trigger(int(self.input_box.text())))

        self.lsl_chunk_checkbox = QtWidgets.QCheckBox('Chunk data')
        self.lsl_chunk_checkbox.setStyleSheet('color: white')
        self.lsl_chunk_checkbox.setChecked(True)

        # Add the button to the layout
        lsl_controls_label = QtWidgets.QLabel("LSL Controls")
        lsl_controls_label.setStyleSheet("color: white; font-size: 20px;")
        lsl_layout.addWidget(lsl_controls_label)
        lsl_layout.addWidget(self.input_box)
        lsl_layout.addWidget(self.trigger_button)
        lsl_layout.addWidget(self.lsl_chunk_checkbox)
        lsl_layout.addWidget(self.lsl_button)

        # Set the layout for the LSL panel
        lsl_panel.setLayout(lsl_layout)
        lsl_panel.setMinimumWidth(250)
        lsl_panel.setMaximumWidth(250)
        lsl_panel.setMaximumHeight(500)
        lsl_panel.setStyleSheet("background-color: #43485E; color: white;")

        return lsl_panel

    def create_osc_panel(self):
        """Create a panel for OSC controls"""
        osc_panel = QtWidgets.QWidget()
        osc_layout = QtWidgets.QVBoxLayout()

        # Create a button to start/stop OSC
        self.osc_address_input = QtWidgets.QLineEdit()
        address_label = QtWidgets.QLabel("OSC Address")
        self.osc_address_input.setText("127.0.0.1")
        self.osc_address_input.setPlaceholderText("OSC Address")
        self.osc_address_input.setFixedWidth(150)
        port_input_label = QtWidgets.QLabel("OSC Listen")
        self.osc_port_input = QtWidgets.QLineEdit()
        self.osc_port_input.setText("8000")
        self.osc_port_input.setPlaceholderText("OSC Listen Port")
        port_output_label = QtWidgets.QLabel("OSC Send")
        self.osc_port_output = QtWidgets.QLineEdit()
        self.osc_port_output.setText("9000")
        self.osc_port_output.setPlaceholderText("OSC Send Port")

        self.osc_button = QtWidgets.QPushButton('Start OSC')
        self.osc_button.setFixedWidth(100)
        self.osc_button.clicked.connect(self.toggle_osc)


        # Add the button to the layout
        osc_controls_label = QtWidgets.QLabel("OSC Controls")
        osc_controls_label.setStyleSheet("color: white; font-size: 20px;")
        osc_layout.addWidget(osc_controls_label)
        osc_layout.addWidget(address_label)
        osc_layout.addWidget(self.osc_address_input)
        osc_layout.addWidget(port_input_label)
        osc_layout.addWidget(self.osc_port_input)
        osc_layout.addWidget(port_output_label)
        osc_layout.addWidget(self.osc_port_output)
        osc_layout.addWidget(self.osc_button)

        # Set the layout for the OSC panel
        osc_panel.setLayout(osc_layout)
        osc_panel.setMinimumWidth(250)
        osc_panel.setMaximumWidth(250)
        osc_panel.setMaximumHeight(500)
        osc_panel.setStyleSheet("background-color: #43485E; color: white;")

        return osc_panel

    def create_parameters_panel(self):
        """Create buttons to interact with the neuroengine"""



        # Start / Stop buttons
        self.start_button = QtWidgets.QPushButton('Stop Plot')
        self.start_button.setFixedWidth(100)
        self.start_button.clicked.connect(lambda: self.toggle_plot())

        # Buttons to plot ROC curve and confusion matrix
        self.roc_button = QtWidgets.QPushButton('Plot ROC')
        self.roc_button.setFixedWidth(100)
        self.roc_button.clicked.connect(lambda: self.classifier.plot_roc_curve())

        self.confusion_button = QtWidgets.QPushButton('Plot CM')
        self.confusion_button.setFixedWidth(100)
        self.confusion_button.clicked.connect(lambda: self.classifier.plot_confusion_matrix())

        # Save data checkbox
        self.save_data_checkbox = QtWidgets.QCheckBox('Save to CSV')
        self.save_data_checkbox.setStyleSheet('color: white')
        self.save_data_checkbox.setChecked(self.save_data)

        # Input box to configure Bandpass filter with checkbox to enable/disable it
        self.bandpass_checkbox = QtWidgets.QCheckBox('Bandpass (Hz)')
        self.bandpass_checkbox.setStyleSheet('color: white')
        self.bandpass_box_low = QtWidgets.QLineEdit()
        self.bandpass_box_low.setPlaceholderText('0')
        self.bandpass_box_low.setText('1')
        self.bandpass_box_low.setMaximumWidth(30)
        self.bandpass_box_high = QtWidgets.QLineEdit()
        self.bandpass_box_high.setPlaceholderText('0')
        self.bandpass_box_high.setText('40')
        self.bandpass_box_high.setMaximumWidth(30)

        # Input box to configure Notch filter with checkbox to enable/disable it and white label
        self.notch_checkbox = QtWidgets.QCheckBox('Notch (Hz)')
        self.notch_checkbox.setStyleSheet('color: white')
        self.notch_box = QtWidgets.QLineEdit()
        self.notch_box.setMaximumWidth(60)  # Set a fixed width for the input box
        self.notch_box.setPlaceholderText('0, 0')
        self.notch_box.setText('50, 60')

        # Create a layout for buttons
        start_save_layout = QtWidgets.QVBoxLayout()
        start_save_layout.addWidget(self.save_data_checkbox)
        start_save_layout.addWidget(self.start_button)

        # Create a layout for the bandpass filter
        bandpass_layout = QtWidgets.QHBoxLayout()
        bandpass_layout.addWidget(self.bandpass_checkbox)
        bandpass_layout.addWidget(self.bandpass_box_low)
        bandpass_layout.addWidget(self.bandpass_box_high)

        # Create a layout for the notch filter
        notch_layout = QtWidgets.QHBoxLayout()
        notch_layout.addWidget(self.notch_checkbox)
        notch_layout.addWidget(self.notch_box)

        # Create a layout for LSL options
        eeg_options_label = QtWidgets.QLabel("EEG Options")
        eeg_options_label.setStyleSheet("color: white; font-size: 20px;")
        eeg_params_layout = QtWidgets.QVBoxLayout()
        eeg_params_layout.addLayout(start_save_layout)

        # Create a vertical layout to contain the notch filter and the button layout
        filters_label = QtWidgets.QLabel("Filters")
        filters_label.setStyleSheet("color: white; font-size: 20px;")
        filter_params_layout = QtWidgets.QVBoxLayout()
        filter_params_layout.addLayout(bandpass_layout)
        filter_params_layout.addLayout(notch_layout)

        # Create a layout for classifier plots
        model_label = QtWidgets.QLabel("Classifier")
        model_label.setStyleSheet("color: white; font-size: 20px;")
        model_params_layout = QtWidgets.QHBoxLayout()
        model_params_layout.addWidget(self.roc_button)
        model_params_layout.addWidget(self.confusion_button)

        # Horizontal layout to contain the classifier buttons
        vertical_container = QtWidgets.QVBoxLayout()
        vertical_container.addWidget(eeg_options_label)
        vertical_container.addLayout(eeg_params_layout)
        vertical_container.addWidget(filters_label)
        vertical_container.addLayout(filter_params_layout)
        vertical_container.addWidget(model_label)
        vertical_container.addLayout(model_params_layout)

        # Create a widget to contain the layout
        parameters = QtWidgets.QWidget()
        parameters.setLayout(vertical_container)
        parameters.setMinimumWidth(250)
        parameters.setMaximumWidth(250)
        parameters.setMaximumHeight(500)
        parameters.setStyleSheet("background-color: #43485E; color: white;")

        return parameters

    def toggle_osc(self):
        if not self.osc_state:
            listen_port = int(self.osc_port_input.text())
            send_port = int(self.osc_port_output.text())
            self.osc_thread = OscStreamThread(listen_port=listen_port, send_port=send_port)
            self.osc_thread.message_received.connect(lambda addr, args: print(f"Received {addr}: {args}"))
            self.osc_thread.start()

            self.osc_button.setText('Stop OSC')
            self.osc_state = True
        else:
            self.osc_thread.stop()
            self.osc_thread = None
            self.osc_button.setText('Start OSC')
            self.osc_state = False

    def toggle_lsl(self):
        """Toggle LSL streaming on and off."""
        if self.lsl_state:
            self.lsl_thread.quit()
            self.lsl_button.setText('Start LSL')
            self.lsl_state = False
        else:
            self.lsl_thread.start()
            self.lsl_button.setText('Stop LSL')
            self.lsl_state = True


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