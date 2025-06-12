import logging

from PyQt5 import QtWidgets

from opencortex.neuroengine.core.stream_engine import StreamEngine
from opencortex.neuroengine.gui.gui_adapter import GUIAdapter


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

        self.stream_engine = StreamEngine(board, config, window_size)
        self.stream_engine.start()
        logging.info("StreamEngine service started")

        self.gui_adapter = GUIAdapter(self.stream_engine)

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
