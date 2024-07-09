import argparse
import logging
from PyQt5 import QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from application.setup_dialog import SetupDialog, retrieve_board_id, retrieve_eeg_devices
from application.streamer import Streamer


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)

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
            selected_device, window_size = dialog.get_data()
            logging.info(f"Selected Device: {selected_device}")
            logging.info(f"Window Size: {window_size} seconds")
            args.serial_number = selected_device
            args.board_id = retrieve_board_id(args.serial_number)
    except BaseException as e:
        logging.info('Impossible to connect device', e)
        return
    window_size = 1 if window_size == 0 else int(window_size)

    params.serial_number = args.serial_number
    board_shim = BoardShim(args.board_id, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(streamer_params=args.streamer_params)
        streamer = Streamer(board_shim, params=params, plot=True, save_data=True, window_size=window_size)
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


if __name__ == '__main__':
    main()
