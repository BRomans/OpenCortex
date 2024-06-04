import logging
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import resolve_stream, StreamInlet


class LSLStreamThread(QThread):
    """Thread to read from an LSL stream and emit new sample data."""

    new_sample = pyqtSignal(object, float)  # Signal to emit new sample data

    def run(self):
        # Resolve an LSL stream named 'MyStream'
        logging.info("Looking for an LSL stream...")
        streams = resolve_stream('type', 'Markers')

        # Create a new inlet to read from the stream
        inlet = StreamInlet(streams[0])

        while True:
            # Pull a new sample from the inlet
            marker, timestamp = inlet.pull_sample()
            logging.info({"got %s at time %s" % (marker[0], timestamp)})
            # Emit the new sample data
            self.new_sample.emit(marker[0], timestamp)
