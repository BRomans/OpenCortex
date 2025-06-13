import os
import argparse
import logging
from sys import platform
from PyQt5 import QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from opencortex.neuroengine.gui.gui_adapter import GUIAdapter
from opencortex.neuroengine.setup_dialog import SetupDialog, retrieve_board_id, retrieve_eeg_devices
from opencortex.neuroengine.streamer_gui import StreamerGUI
from opencortex.neuroengine.core.stream_engine import StreamEngine, HeadlessStreamEngine

logging_levels = {0: logging.NOTSET, 1: logging.DEBUG, 2: logging.INFO, 3: logging.WARNING, 4: logging.ERROR,
                  5: logging.CRITICAL}

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "config", "Default.yaml")

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


def run_headless():
    """Run OpenCortex in headless mode (no GUI)"""
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    # Add all the existing arguments
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
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

    # New headless-specific arguments
    parser.add_argument('--headless', action='store_true', help='run in headless mode (no GUI)')
    parser.add_argument('--config-file', type=str, help='config file path', required=False,
                        default='config/Default.yaml')
    parser.add_argument('--window-size', type=int, help='window size in seconds', required=False, default=1)
    parser.add_argument('--log-file', type=str, help='log file for headless mode', required=False, default=None)

    args = parser.parse_args()

    # Load configuration
    import yaml
    config_file = os.path.join(base_dir, args.config_file)
    if not os.path.exists(config_file):
        logging.warning(f'Config file {config_file} does not exist, using default config')
        config_file = config_path

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Set up BrainFlow parameters
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
        # Connect to board
        board_shim = BoardShim(args.board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream(streamer_params=args.streamer_params)

        # Create headless StreamEngine
        engine = HeadlessStreamEngine(board_shim, config, args.window_size, args.log_file)

        # Add monitoring callbacks
        def data_monitor(data):
            logging.info(f"Data update: {len(data.quality_scores)} channels, "
                         f"trigger: {data.trigger}, quality: {data.quality_scores}")

        def event_monitor(event_type, event_data):
            logging.info(f"Event: {event_type} - {event_data}")

        engine.register_data_callback(data_monitor)
        engine.register_event_callback(event_monitor)

        logging.info("Starting OpenCortex in headless mode...")
        logging.info(f"Board: {board_shim.get_device_name(args.board_id)}")
        logging.info(f"Sampling rate: {BoardShim.get_sampling_rate(args.board_id)} Hz")
        logging.info(f"EEG channels: {BoardShim.get_eeg_names(args.board_id)}")
        logging.info("Press Ctrl+C to stop...")

        # Run forever
        engine.run_forever()

    except Exception as e:
        logging.error(f'Error in headless mode: {e}')
    finally:
        if 'board_shim' in locals() and board_shim.is_prepared():
            board_shim.stop_stream()
            board_shim.release_session()
        logging.info('Headless mode ended')


def run_gui():
    """Run OpenCortex with GUI (existing functionality)"""
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    # Original arguments
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
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

    # Add service mode option
    parser.add_argument('--service-mode', action='store_true',
                        help='use new service-based architecture (recommended)')

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
        else:
            logging.info("Setup dialog cancelled")
            return

    except BaseException as e:
        logging.info('Impossible to connect device', e)
        return

    window_size = 1 if window_size == 0 else int(window_size)
    com_ports = get_com_ports()
    params.serial_number = args.serial_number
    config_file = os.path.join(base_dir, 'config', args.config_file)

    if not os.path.exists(config_file):
        logging.warning(f'Config file {config_file} does not exist, using default config')
        config_file = config_path
    else:
        logging.info(f'Loaded config file: {config_file}')

    board_shim = None
    streamer = None

    try:
        # Connect to board (existing logic)
        if args.board_id in open_bci_ids:
            for com_port in com_ports:
                try:
                    params.serial_port = com_port
                    board_shim = BoardShim(args.board_id, params)
                    board_shim.prepare_session()
                    board_shim.start_stream(streamer_params=args.streamer_params)
                    logging.info(f"Connected to {com_port}")
                    break
                except BaseException:
                    logging.warning(f'Could not connect to port {com_port}, trying next one')

            if board_shim is None:
                raise Exception("Could not connect to any COM port")
        else:
            board_shim = BoardShim(args.board_id, params)
            board_shim.prepare_session()
            board_shim.start_stream(streamer_params=args.streamer_params)

        # Choose architecture based on flag
        if args.service_mode:
            logging.info("Starting with new service-based architecture")
            streamer = StreamerGUIService(board_shim, params=params,
                                          window_size=window_size, config_file=config_file)
        else:
            logging.info("Starting with legacy architecture")
            streamer = StreamerGUI(board_shim, params=params,
                                   window_size=window_size, config_file=config_file)

    except BaseException as e:
        logging.warning('Exception', exc_info=True)
        logging.error(f'Could not prepare session: {e}')
    finally:
        logging.info('Application ending')
        if board_shim and board_shim.is_prepared():
            logging.info('Releasing session')
            try:
                if streamer:
                    streamer.quit()
            except BaseException:
                logging.warning('Streaming has already been stopped')
            board_shim.stop_stream()
            board_shim.release_session()


def run():
    """Main entry point - decide between GUI and headless mode"""
    import sys

    # Check for headless flag early
    if '--headless' in sys.argv:
        run_headless()
    else:
        run_gui()

class StreamerGUIService:
    """
    New GUI class that uses StreamEngine service.
    This will eventually replace the original StreamerGUI.
    """

    def __init__(self, board, params, window_size=1, config_file='Default.yaml'):
        import yaml

        # Load configuration
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        logging.info("Initializing StreamerGUI with service architecture")

        # ============ CREATE STREAMENGINE SERVICE ============
        self.stream_engine = StreamEngine(board, config, window_size)
        self.stream_engine.start()
        logging.info("StreamEngine service started")

        # ============ CREATE GUI ADAPTER ============
        self.gui_adapter = GUIAdapter(self.stream_engine)

        # ============ INITIALIZE EXISTING GUI ============
        # For now, we'll create a minimal GUI that just shows status
        # Later, we'll move the full GUI logic here

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        self.create_simple_gui()

        # Connect adapter signals
        self.gui_adapter.data_updated.connect(self.on_data_updated)
        self.gui_adapter.prediction_ready.connect(self.on_prediction_ready)
        self.gui_adapter.classifier_ready.connect(self.on_classifier_ready)
        self.gui_adapter.error_occurred.connect(self.on_error)

        # Start the GUI
        self.app.exec_()

    def create_simple_gui(self):
        """Create a simple status GUI for testing"""
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle('OpenCortex - Service Mode')
        self.window.resize(400, 300)

        layout = QtWidgets.QVBoxLayout()

        # Status label
        self.status_label = QtWidgets.QLabel('StreamEngine Status: Starting...')
        layout.addWidget(self.status_label)

        # Data counter
        self.data_counter_label = QtWidgets.QLabel('Data updates: 0')
        layout.addWidget(self.data_counter_label)
        self.data_count = 0

        # Engine info
        status = self.stream_engine.get_status()
        info_text = f"""
        Board ID: {status['board_id']}
        Sampling Rate: {status['sampling_rate']} Hz
        EEG Channels: {status['eeg_channels']}
        """
        self.info_label = QtWidgets.QLabel(info_text)
        layout.addWidget(self.info_label)

        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.trigger_button = QtWidgets.QPushButton('Send Trigger')
        self.trigger_button.clicked.connect(lambda: self.gui_adapter.send_trigger(1))
        button_layout.addWidget(self.trigger_button)

        self.inference_button = QtWidgets.QPushButton('Toggle Inference')
        self.inference_button.clicked.connect(lambda: self.gui_adapter.set_inference_mode())
        button_layout.addWidget(self.inference_button)

        layout.addLayout(button_layout)

        # Quit button
        quit_button = QtWidgets.QPushButton('Quit')
        quit_button.clicked.connect(self.quit)
        layout.addWidget(quit_button)

        self.window.setLayout(layout)
        self.window.show()

    def on_data_updated(self, stream_data):
        """Handle data updates from StreamEngine"""
        self.data_count += 1
        self.data_counter_label.setText(f'Data updates: {self.data_count}')

        # Update status
        status = self.stream_engine.get_status()
        self.status_label.setText(
            f"StreamEngine Status: Running ({'Inference' if status['inference_mode'] else 'Recording'})")

    def on_prediction_ready(self, prediction):
        """Handle prediction results"""
        logging.info(f"GUI received prediction: {prediction}")
        # Could show prediction in GUI here

    def on_classifier_ready(self):
        """Handle classifier ready"""
        self.status_label.setText("StreamEngine Status: Classifier Ready")
        logging.info("GUI notified: Classifier ready")

    def on_error(self, error_msg):
        """Handle errors"""
        logging.error(f"GUI received error: {error_msg}")
        # Could show error dialog here

    def quit(self):
        """Clean shutdown"""
        logging.info("Shutting down StreamerGUI service mode")
        self.stream_engine.stop()
        self.gui_adapter.cleanup()
        self.app.quit()


if __name__ == '__main__':
    run()

