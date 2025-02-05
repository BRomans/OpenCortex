import argparse
import logging
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

eno_channels = ["A1", "C3", "C4", "A2"]


class Graph:
    def __init__(self, board_shim):
        self.board_shim = board_shim  # Assuming board_shim is already initialized
        self.board_id = board_shim.get_board_id()
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 1000
        self.window_size = 1
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication([])

        # Open CSV file
        self.file = open('enodata.csv', 'w')
        # Write header taking elements from the list of EEG channels
        self.file.write('Sample\t')
        for channel in eno_channels:
            self.file.write(str(channel) + '\t')
        self.file.write('Time\tTrigger\n')

        # Create a window
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot', size=(800, 600))
        self.win.setWindowTitle('BrainFlow Plot')
        self.win.show()

        self._init_timeseries()

        # Start the PyQt event loop
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        self.app.exec_()

    def _init_timeseries(self):
        color_thresholds = [(-150, -50, 'yellow'), (-50, 50, 'green'), (50, 150, 'yellow')]
        self.plots = list()
        self.curves = list()
        ylim = (-500, 500)
        for i, channel_name in enumerate(self.eeg_channels):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', True)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', True)
            p.setMenuEnabled('bottom', False)
            p.setYRange(*ylim)
            p.setTitle(eno_channels[i])  # Set title to channel name for each plot
            p.setLabel('left', text='Amplitude (uV)')  # Label for y-axis
            p.setLabel('bottom', text='Time (s)')  # Label for x-axis
            self.plots.append(p)
            curve = p.plot(pen='y')  # Set a specific color for each curve
            self.curves.append(curve)
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
        curve = p.plot(pen='y')
        self.curves.append(curve)

    def update(self):
        self.board_shim.insert_marker(1)
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            self.curves[count].setData(data[channel].tolist())
        # plot trigger channel
        trigger = data[-1]
        self.curves[-1].setData(trigger.tolist())

        # Write data to file

        DataFilter.write_file(data, 'enodata.csv', 'a')
        # for row in data.T:
        # data_str = ','.join(map(str,  list(row))) + '\n'
        # self.file.write(data_str)
        self.app.processEvents()


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
    parser.add_argument('--neuroengine-params', type=str, help='neuroengine params', required=False, default='')
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

    board_shim = BoardShim(args.board_id, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000, args.streamer_params)
        Graph(board_shim)
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
