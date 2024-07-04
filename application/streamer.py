import threading
import time
import matplotlib
import numpy as np
import logging
import pyqtgraph as pg
import os
import pylsl
from PyQt5 import QtWidgets, QtCore
from application.classifier import Classifier
from application.lsl_stream import LSLStreamThread
from utils.layouts import layouts
from pyqtgraph import ScatterPlotItem, mkBrush
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from concurrent.futures import ThreadPoolExecutor

matplotlib.use("Qt5Agg")

# 16 Color ascii codes for the 16 EEG channels
colors = ["blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray",
          "cyan", "magenta", "lime", "teal", "lavender", "turquoise", "maroon", "olive"]


def write_header(file, board_id):
    for column in layouts[board_id]["header"]:
        file.write(str(column) + '\t')
    file.write('\n')


class Streamer:

    def __init__(self, board, params, plot=True, save_data=True, window_size=1, update_speed_ms=1000, model='LDA'):
        time.sleep(window_size)  # Wait for the board to be ready

        self.is_streaming = True
        self.params = params
        self.initial_ts = time.time()
        logging.info("Searching for devices...")
        self.board = board
        self.board_id = self.board.get_board_id()
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.sample_counter = 0
        self.color_thresholds = [(-150, -50, 'yellow'), (-50, 50, 'green'), (50, 150, 'yellow')]
        self.update_speed_ms = update_speed_ms
        self.window_size = window_size
        self.plot = plot
        self.save_data = save_data
        self.num_points = self.window_size * self.sampling_rate
        self.filtered_eeg = np.zeros((len(self.eeg_channels) + 1, self.num_points))
        logging.info(f"Connected to {self.board.get_device_name(self.board.get_board_id())}")

        # Initialize the classifier in a new thread
        self.classifier = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        if model is not None:
            self.model = model
            self.over_sample = False
            self.classifier_thread = threading.Thread(target=self.init_classifier)
            self.classifier_thread.start()

        self.app = QtWidgets.QApplication([])

        # Calculate time interval for prediction
        self.nclasses = 3
        self.on_time = 150 # ms
        self.off_time = (self.on_time * (self.nclasses - 1))
        logging.info(f"Off time: {self.off_time} ms")
        self.prediction_interval = int(self.on_time + self.off_time)
        logging.info(f"Prediction interval: {self.prediction_interval} ms")
        # calculate how many datapoints based on the sampling rate
        self.prediction_datapoints = int(self.prediction_interval * self.sampling_rate / 1000)
        logging.info(f"Prediction interval in datapoints: {self.prediction_datapoints}")

        logging.info("Looking for an LSL stream...")
        # Connect to the LSL stream threads
        self.prediction_timer = QtCore.QTimer()
        self.prediction_timer.timeout.connect(self.predict_class)
        self.lsl_thread = LSLStreamThread()
        self.lsl_thread.new_sample.connect(self.write_trigger)
        self.lsl_thread.set_train_start.connect(self.set_train_start)
        self.lsl_thread.start_train.connect(self.train_classifier)
        self.lsl_thread.start_predicting.connect(self.start_prediction)
        self.lsl_thread.stop_predicting.connect(self.stop_prediction)
        self.lsl_thread.start()

        if plot:
            # Create a window
            self.win = pg.GraphicsLayoutWidget(title='EEG Plot', size=(1200, 800))
            self.win.setWindowTitle('EEG Plot')
            self.win.show()
            self._init_timeseries()
            self.create_buttons()

        # Start the PyQt event loop to fetch the raw data
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        self.start_lsl_eeg_stream()
        self.start_lsl_prediction_stream()
        self.app.exec_()

    def create_buttons(self):
        """Create buttons to interact with the streamer"""

        # Button to write trigger and input box to specify the trigger value
        self.input_box = QtWidgets.QLineEdit()
        self.input_box.setFixedWidth(100)  # Set a fixed width for the input box
        self.input_box.setPlaceholderText('Trigger value')
        self.input_box.setText('1')

        self.trigger_button = QtWidgets.QPushButton('Send Trigger')
        self.trigger_button.setFixedWidth(100)  # Set a fixed width for the button
        self.trigger_button.clicked.connect(lambda: self.write_trigger(int(self.input_box.text())))

        # Start / Stop buttons
        self.start_button = QtWidgets.QPushButton('Stop')
        self.start_button.setFixedWidth(100)
        self.start_button.clicked.connect(lambda: self.toggle_stream())

        # Buttons to plot ROC curve and confusion matrix
        self.roc_button = QtWidgets.QPushButton('Plot ROC')
        self.roc_button.setFixedWidth(100)
        self.roc_button.clicked.connect(lambda: self.classifier.plot_roc_curve())

        self.confusion_button = QtWidgets.QPushButton('Plot CM')
        self.confusion_button.setFixedWidth(100)
        self.confusion_button.clicked.connect(lambda: self.classifier.plot_confusion_matrix())

        # Save data checkbox
        self.save_data_checkbox = QtWidgets.QCheckBox('Save data to file')
        self.save_data_checkbox.setStyleSheet('color: white')
        self.save_data_checkbox.setChecked(self.save_data)

        # Input box to configure Bandpass filter with checkbox to enable/disable it
        self.bandpass_checkbox = QtWidgets.QCheckBox('Bandpass filter frequencies (Hz)')
        self.bandpass_checkbox.setStyleSheet('color: white')
        self.bandpass_box_low = QtWidgets.QLineEdit()
        self.bandpass_box_low.setPlaceholderText('0')
        self.bandpass_box_low.setText('1')
        self.bandpass_box_low.setFixedWidth(25)
        self.bandpass_box_high = QtWidgets.QLineEdit()
        self.bandpass_box_high.setPlaceholderText('0')
        self.bandpass_box_high.setText('40')
        self.bandpass_box_high.setFixedWidth(25)

        # Input box to configure Notch filter with checkbox to enable/disable it and white label
        self.notch_checkbox = QtWidgets.QCheckBox('Notch filter frequencies (Hz)')
        self.notch_checkbox.setStyleSheet('color: white')
        self.notch_box = QtWidgets.QLineEdit()
        self.notch_box.setFixedWidth(56)  # Set a fixed width for the input box
        self.notch_box.setPlaceholderText('0, 0')
        self.notch_box.setText('50, 60')

        # Create a layout for buttons
        start_save_layout = QtWidgets.QHBoxLayout()
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

        # Create a vertical layout to contain the notch filter and the button layout
        left_side_label = QtWidgets.QLabel("Filters")
        left_side_label.setStyleSheet("color: white; font-size: 20px;")
        left_side_layout = QtWidgets.QVBoxLayout()
        left_side_layout.addWidget(left_side_label)
        left_side_layout.addLayout(bandpass_layout)
        left_side_layout.addLayout(notch_layout)
        left_side_layout.addLayout(start_save_layout)

        # Create a center layout for trigger button
        center_label = QtWidgets.QLabel("Markers")
        center_label.setStyleSheet("color: white; size: 20px;")
        center_layout = QtWidgets.QVBoxLayout()
        center_layout.addWidget(center_label)
        center_layout.addWidget(self.input_box)
        center_layout.addWidget(self.trigger_button)
        center_layout.addStretch()

        # Create a layout for classifier plots
        right_side_label = QtWidgets.QLabel("Classifier")
        right_side_label.setStyleSheet("color: white; font-size: 20px;")
        right_side_layout = QtWidgets.QVBoxLayout()
        right_side_layout.addWidget(right_side_label)
        right_side_layout.addWidget(self.roc_button)
        right_side_layout.addWidget(self.confusion_button)
        right_side_layout.addStretch()

        # Horizontal layout to contain the classifier buttons
        horizontal_container = QtWidgets.QHBoxLayout()
        horizontal_container.addLayout(left_side_layout)
        horizontal_container.addLayout(center_layout)
        horizontal_container.addLayout(right_side_layout)

        # Create a widget to contain the layout
        button_widget = QtWidgets.QWidget()
        button_widget.setLayout(horizontal_container)

        # Create a layout for the main window
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(button_widget,
                              alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)  # Align to bottom right

        # Set the main layout for the window
        self.win.setLayout(main_layout)

    def _init_timeseries(self):
        """Initialize the timeseries plots for the EEG channels"""
        self.plots = list()
        self.curves = list()
        self.quality_indicators = []
        for i, channel in enumerate(self.eeg_channels):
            if i < len(self.eeg_channels) / 2:
                row = i
                col = 0
            else:
                row = i - len(self.eeg_channels) / 2
                col = 1
            p = self.win.addPlot(row=int(row), col=col)
            # Increase size of the plot
            p.showAxis('left', True)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setMinimumWidth(400)
            p.setTitle(layouts[self.board_id]["channels"][i])  # Set title to channel name for each plot
            p.setLabel('left', text='Amplitude (uV) ')  # Label for y-axis
            p.setLabel('bottom', text='Time (s)')  # Label for x-axis
            self.plots.append(p)
            curve = p.plot(pen=colors[i])  # Set a specific color for each curve
            self.curves.append(curve)

            # Create a scatter plot item for the quality indicator
            scatter = ScatterPlotItem(size=20, brush=mkBrush('green'))
            p.addItem(scatter)
            self.quality_indicators.append(scatter)

        # plot trigger channel
        p = self.win.addPlot(row=len(self.eeg_channels), col=0)
        p.showAxis('left', True)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', True)
        p.setMenuEnabled('bottom', False)
        p.setYRange(0, 5)
        p.setTitle('Trigger')
        p.setLabel('left', text='Trigger')
        p.setLabel('bottom', text='Time (s)')
        self.plots.append(p)
        curve = p.plot(pen='red')
        self.curves.append(curve)

    def update(self):
        """ Update the plot with new data"""
        if self.is_streaming:
            if self.window_size == 0:
                raise ValueError("Window size cannot be zero")
            data = self.board.get_current_board_data(num_samples=self.num_points)
            start_eeg = layouts[self.board_id]["eeg_start"]
            end_eeg = layouts[self.board_id]["eeg_end"]
            eeg = data[start_eeg:end_eeg]
            self.push_lsl_eeg_chunk(data)
            self.sample_counter += 1
            # logging.info(f"Pulling sample {self.sample_counter}: {data.shape}")
            if self.plot:
                for count, channel in enumerate(self.eeg_channels):
                    ch_data = eeg[count]
                    # plot timeseries
                    DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
                    try:
                        if self.bandpass_checkbox.isChecked():
                            low = float(self.bandpass_box_low.text())
                            high = float(self.bandpass_box_high.text())
                            DataFilter.perform_bandpass(ch_data, self.sampling_rate, low, high, 4,
                                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                    except ValueError:
                        logging.error("Invalid frequency value")
                    try:
                        if self.notch_checkbox.isChecked():
                            freqs = self.notch_box.text().split(',')
                            for freq in freqs:
                                start_freq = float(freq) - 2.0
                                end_freq = float(freq) + 2.0
                                DataFilter.perform_bandstop(ch_data, self.sampling_rate, start_freq, end_freq, 4,
                                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                    except ValueError:
                        logging.error("Invalid frequency value")
                    self.curves[count].setData(ch_data)
                    self.filtered_eeg[count] = ch_data
                    # Rescale the plot
                    self.plots[count].setYRange(np.min(ch_data), np.max(ch_data))

                # plot trigger channel
                trigger = data[-1]
                self.filtered_eeg[-1] = trigger
                # log the filtered eeg
                self.curves[-1].setData(trigger.tolist())
                self.app.processEvents()
            self.update_quality_indicators(data)

    def start_lsl_eeg_stream(self, stream_name='myeeg', type='EEG'):
        """
        Start an LSL stream for the EEG data
        :param stream_name: str, name of the LSL stream
        :param type: str, type of the LSL stream
        """
        try:
            info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=len(self.eeg_channels) + 1,
                                    nominal_srate=self.sampling_rate, channel_format='float32',
                                    source_id=self.board.get_device_name(self.board_id))
            # Add channel names
            chs = info.desc().append_child("channels")
            for ch in layouts[self.board_id]["channels"]:
                chs.append_child("channel").append_child_value("name", ch)
            chs.append_child("channel").append_child_value("name", "Trigger")
            self.eeg_outlet = pylsl.StreamOutlet(info)
        except Exception as e:
            logging.error(f"Error starting LSL stream: {e}")

    def start_lsl_prediction_stream(self, stream_name='mypredictions', type='Markers'):
        """
        Start an LSL stream for the prediction data
        :param stream_name: str, name of the LSL stream
        :param type: str, type of the LSL stream
        """
        try:
            info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=1,
                                    nominal_srate=self.sampling_rate, channel_format='float32',
                                    source_id=self.board.get_device_name(self.board_id))
            self.prediction_outlet = pylsl.StreamOutlet(info)
        except Exception as e:
            logging.error(f"Error starting LSL stream: {e}")

    def push_lsl_eeg_chunk(self, data):
        """
        Push a chunk of data to the LSL stream
        :param data: numpy array of shape (n_channels, n_samples)
        """
        try:
            # Get EEG and Trigger from data and push it to LSL
            start_eeg = layouts[self.board_id]["eeg_start"]
            end_eeg = layouts[self.board_id]["eeg_end"]
            eeg = data[start_eeg:end_eeg]
            trigger = data[-1]
            # Horizontal stack EEG and Trigger
            eeg = np.concatenate((eeg, trigger.reshape(1, len(trigger))), axis=0)
            ts_channel = self.board.get_timestamp_channel(self.board_id)
            ts = data[ts_channel]
            ts_to_lsl_offset = time.time() - pylsl.local_clock()
            # Get only the seconds part of the timestamp
            ts = ts - ts_to_lsl_offset
            self.eeg_outlet.push_chunk(eeg.T.tolist(), ts)
            logging.debug(f"Pushed sample {self.sample_counter} to LSL")
        except Exception as e:
            logging.error(f"Error pushing chunk to LSL: {e}")

    def push_lsl_eeg_packet(self, data):
        """
        Push a packet of data to the LSL stream
        :param data: numpy array of shape (n_channels, n_samples)
        """
        # Get EEG and Trigger from data and push it to LSL
        try:
            start_eeg = layouts[self.board_id]["eeg_start"]
            end_eeg = layouts[self.board_id]["eeg_end"]
            eeg = data[start_eeg:end_eeg]
            trigger = data[-1]
            # Horizontal stack EEG and Trigger
            eeg = np.concatenate((eeg, trigger.reshape(1, len(trigger))), axis=0)
            ts_channel = self.board.get_timestamp_channel(self.board_id)
            ts = data[ts_channel]
            ts_to_lsl_offset = time.time() - pylsl.local_clock()
            ts = ts - ts_to_lsl_offset
            for i in range(eeg.shape[1]):
                sample = eeg[:, i]
                self.eeg_outlet.push_sample(sample.tolist(), ts[i])
        except Exception as e:
            logging.error(f"Error pushing sample to LSL: {e}")

    def push_lsl_prediction(self, prediction):
        """
        Push a prediction to the LSL stream
        """
        try:
            self.prediction_outlet.push_sample([prediction])
            logging.info("Pushed prediction to LSL", prediction)
        except Exception as e:
            logging.error(f"Error pushing prediction to LSL: {e}")

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
        if trigger == '':
            logging.error("Trigger value cannot be empty")
            return
        if timestamp == 0:
            timestamp = time.time()
        self.board.insert_marker(int(trigger))

    def init_classifier(self):
        """ Initialize the classifier """
        self.classifier = Classifier(model=self.model, board_id=self.board_id)

    def set_train_start(self):
        """" Set the start of the training"""
        self.start_training_time = time.time()

    def train_classifier(self):
        """ Train the classifier"""
        end_training_time = time.time()
        training_length = end_training_time - self.start_training_time + 1
        training_interval = int(training_length * self.sampling_rate)
        data = self.board.get_current_board_data(training_interval)
        self.classifier.train(data, oversample=self.over_sample)

    def start_prediction(self):
        """Start the prediction timer."""
        self.prediction_timer.start(self.prediction_interval)

    def stop_prediction(self):
        """Stop the prediction timer."""
        self.prediction_timer.stop()

    def _predict_class(self, data):
        """Internal method to predict the class of the data."""
        try:
            output = self.classifier.predict(data, proba=True)
            self.push_lsl_prediction(output)
            logging.info(f"Predicted class: {output}")
        except Exception as e:
            logging.error(f"Error predicting class: {e}")

    def predict_class(self):
        """Predict the class of the data."""
        try:
            data = self.filtered_eeg
            self.executor.submit(self._predict_class, data)
        except Exception as e:
            logging.error(f"Error starting prediction task: {e}")

    def update_quality_indicators(self, sample):
        """ Update the quality indicators for each channel"""
        eeg_start = layouts[self.board_id]["eeg_start"]
        eeg_end = layouts[self.board_id]["eeg_end"]
        eeg = sample[eeg_start:eeg_end]
        amplitudes = []
        q_colors = []
        for i in range(len(eeg)):
            amplitude_data = eeg[i]  # get the data for the i-th channel
            color, amplitude = self.get_channel_quality(amplitude_data)
            q_colors.append(color)
            amplitudes.append(np.round(amplitude, 2))
            # Update the scatter plot item with the new color
            self.quality_indicators[i].setBrush(mkBrush(color))
            self.quality_indicators[i].setData([-1], [0])  # Position the circle at (0, 0)
        logging.debug(f"Qualities: {amplitudes} {q_colors}")

    def get_channel_quality(self, eeg, threshold=75):
        """ Get the quality of the EEG channel based on the amplitude"""
        amplitude = np.percentile(eeg, threshold)
        color = 'red'
        for low, high, color_name in self.color_thresholds:
            if low <= amplitude <= high:
                color = color_name
                break
        return color, amplitude

    def toggle_stream(self):
        """ Start or stop the streaming of data"""
        if self.is_streaming:
            self.board.stop_stream()
            self.start_button.setText('Start')
            self.is_streaming = False
        else:
            self.board.start_stream()
            self.start_button.setText('Stop')
            self.is_streaming = True

    def quit(self):
        """ Quit the application, join the threads and stop the streaming"""
        if self.save_data_checkbox.isChecked():
            self.export_file()
        self.lsl_thread.quit()
        self.classifier_thread.join()
        self.board.stop_stream()