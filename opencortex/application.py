import os
import argparse
import logging
from sys import platform
from PyQt5 import QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from opencortex.neuroengine.setup_dialog import SetupDialog, retrieve_board_id, retrieve_eeg_devices
from opencortex.neuroengine.opencortex_ng import OpenCortexEngine

logging_levels = {0: logging.NOTSET, 1: logging.DEBUG, 2: logging.INFO, 3: logging.WARNING, 4: logging.ERROR,
                  5: logging.CRITICAL}

base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the script
config_path = os.path.join(base_dir, "default_config.yaml")

open_bci_ids = [BoardIds.CYTON_BOARD, BoardIds.CYTON_DAISY_BOARD, BoardIds.CYTON_DAISY_WIFI_BOARD,
                BoardIds.CYTON_WIFI_BOARD, BoardIds.GANGLION_BOARD, BoardIds.GANGLION_WIFI_BOARD,
                BoardIds.GANGLION_NATIVE_BOARD]


def get_com_ports():
    com_ports = []
    if platform == 'win32':
        com_ports = ['COM' + str(i) for i in range(6)]
    elif platform == 'linux':
        com_ports = ['/dev/ttyUSB' + str(i) for i in range(6)]
    elif platform == 'darwin':
        com_ports = ['/dev/tty.usbserial-' + str(i) for i in range(6)]
    return com_ports


def run():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=6789)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM3')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.NO_BOARD)
    parser.add_argument('--unicorn-index', type=int, help='index of the unicorn device', required=False, default=0)
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
    try:
        devices = retrieve_eeg_devices()
        win = QtWidgets.QApplication([])

        dialog = SetupDialog(devices)
        window_size = 0
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_device, window_size, log_level, config_file = dialog.get_data()
            logging.info(f"Selected Device: {selected_device}")
            logging.info(f"Window Size: {window_size} seconds")
            logging.info(f"Logging set to level: {log_level}")
            logging.getLogger().setLevel(logging_levels[log_level])
            args.serial_number = selected_device
            args.board_id = retrieve_board_id(args.serial_number)
            args.config_file = config_file
    except BaseException as e:
        logging.info('Impossible to connect device', e)
        return
    window_size = 1 if window_size == 0 else int(window_size)
    com_ports = get_com_ports()
    params.serial_number = args.serial_number
    board_shim = BoardShim(args.board_id, params)
    config_file = args.config_file
    if not os.path.exists(config_file):
        logging.warning(f'Config file {config_file} does not exist, using default config')
        config_file = config_path
    else:
        logging.info(f'Loaded config file: {config_file}')
    try:
        if board_shim.board_id in open_bci_ids:
            for com_port in com_ports:
                try:
                    params.serial_port = com_port
                    board_shim = BoardShim(args.board_id, params)
                    board_shim.prepare_session()
                    board_shim.start_stream(streamer_params=args.streamer_params)
                    streamer = OpenCortexEngine(board_shim, params=params, window_size=window_size, config_file=config_file)
                    break
                except BaseException:
                    logging.warning(f'Could not connect to port {com_port}, trying next one')
        else:
            board_shim.prepare_session()
            board_shim.start_stream(streamer_params=args.streamer_params)
            streamer = OpenCortexEngine(board_shim, params=params, window_size=window_size, config_file=config_file)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            try:
                streamer.quit()
            except BaseException:
                logging.warning('Streaming has already been stopped')
            board_shim.release_session()
