import re
import time
import bluetooth
import argparse
import matplotlib
import numpy as np
import logging
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from utils.layouts import layouts
from pyqtgraph import ScatterPlotItem, mkBrush
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

matplotlib.use("Qt5Agg")

# Color codes for matplotlib
colors = ["blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray", "cyan", "magenta", ]


def retrieve_unicorn_devices():
    saved_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
    unicorn_devices = filter(lambda x: re.search(r'UN-\d{4}.\d{2}.\d{2}', x[1]), saved_devices)
    return list(unicorn_devices)


def write_header(file, board_id):
    for column in layouts[board_id]["header"]:
        file.write(str(column) + '\t')
    file.write('\n')


class Streamer:

    def __init__(self, board, params, plot=True, save_data=True, window_size=1, update_speed_ms=1000):
        time.sleep(1)  # Wait for the board to be ready
        self.params = params
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
        logging.info(f"Connected to {self.board.get_device_name(self.board.get_board_id())}")
        self.app = QtWidgets.QApplication([])

        if save_data:
            # Open CSV file
            with open('session.csv', 'w') as self.file:
                # Write header taking elements from the list of EEG channels
                write_header(self.file, self.board_id)

        if plot:
            # Create a window
            self.win = pg.GraphicsLayoutWidget(title='EEG Plot', size=(1200, 800))
            self.win.setWindowTitle('EEG Plot')
            self.win.show()
            self._init_timeseries()
            self.create_buttons()

        # Start the PyQt event loop
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        self.app.exec_()

    def create_buttons(self):
        # Button to write trigger and input box to specify the trigger value
        self.input_box = QtWidgets.QLineEdit()
        self.input_box.setFixedWidth(100)  # Set a fixed width for the input box
        self.input_box.setPlaceholderText('Trigger value')

        self.trigger_button = QtWidgets.QPushButton('Send Trigger')
        self.trigger_button.setFixedWidth(100)  # Set a fixed width for the button
        self.trigger_button.clicked.connect(lambda: self.write_trigger(int(self.input_box.text())))

        # Create a layout for the button and input box
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)  # Add a stretchable space to push buttons to the right
        button_layout.addWidget(self.trigger_button)
        button_layout.addWidget(self.input_box)

        # Create a widget to contain the layout
        button_widget = QtWidgets.QWidget()
        button_widget.setLayout(button_layout)

        # Create a layout for the main window
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(button_widget,
                              alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)  # Align to bottom right

        # Set the main layout for the window
        self.win.setLayout(main_layout)

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        self.quality_indicators = []
        ylim = (-500, 500)

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
            # p.setYRange(*ylim)
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
        if self.window_size == 0:
            raise ValueError("Window size cannot be zero")
        if self.window_size == 1:
            data = self.board.get_board_data(num_samples=self.num_points)
        else:
            data = self.board.get_current_board_data(num_samples=self.num_points)
        logging.info(f"Pulling sample {self.sample_counter}: {data.shape}")
        if self.plot:
            for count, channel in enumerate(self.eeg_channels):
                ch_data = data[channel]
                # plot timeseries
                DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(ch_data, self.sampling_rate, 3.0, 45.0, 4,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.perform_bandstop(ch_data, self.sampling_rate, 48.0, 52.0, 4,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                DataFilter.perform_bandstop(ch_data, self.sampling_rate, 58.0, 62.0, 4,
                                            FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

                self.curves[count].setData(ch_data)
                # Rescale the plot
                self.plots[count].setYRange(np.min(ch_data), np.max(ch_data))

            # plot trigger channel
            trigger = data[-1]
            self.curves[-1].setData(trigger.tolist())
            self.app.processEvents()
        if self.save_data:
            if self.board_id == BoardIds.UNICORN_BOARD:
                DataFilter.write_file(data, 'session.csv', 'a')
            else:
                DataFilter.write_file(data, 'session.csv', 'a')
            self.sample_counter += 1
        self.update_quality_indicators(data)

    def write_trigger(self, trigger=1):
        self.board.insert_marker(trigger)
        logging.info(f"Trigger {trigger} written at {time.time()}")

    def update_quality_indicators(self, sample):
        eeg_start = layouts[self.board_id]["eeg_start"]
        eeg_end = layouts[self.board_id]["eeg_end"]
        eeg = sample[eeg_start:eeg_end]
        amplitudes = []
        q_colors = []
        for i in range(len(eeg)):
            amplitude_data = eeg[i]  # get the data for the i-th channel
            avg_amplitude = np.percentile(amplitude_data, 75)  # Calculate average amplitude
            amplitudes.append(avg_amplitude)
            # Determine the color based on the average amplitude
            color = 'red'
            for low, high, color_name in self.color_thresholds:
                if low <= avg_amplitude <= high:
                    color = color_name
                    break
            q_colors.append(color)
            # Update the scatter plot item with the new color
            self.quality_indicators[i].setBrush(mkBrush(color))
            self.quality_indicators[i].setData([-1], [0])  # Position the circle at (0, 0)
        logging.info(f"Qualities: {amplitudes} {q_colors}")

    def get_channel_amplitude(self, sample, channel):
        channel_data = sample[channel]
        return


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.ENOPHONE_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board
    if args.board_id == BoardIds.UNICORN_BOARD:
        args.serial_number = retrieve_unicorn_devices()[2][1]
    params.serial_number = args.serial_number
    board_shim = BoardShim(args.board_id, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(streamer_params=args.streamer_params)
        Streamer(board_shim, params=params, plot=True, save_data=True, window_size=1, update_speed_ms=1000)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:

        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.stop_stream()
            board_shim.release_session()


if __name__ == '__main__':
    main()
