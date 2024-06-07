import logging
import pytz
import time
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import resolve_stream, StreamInlet


class LSLStreamThread(QThread):
    """Thread to read from an LSL stream and emit new sample data."""

    new_sample = pyqtSignal(object, float)  # Signal to emit new sample data
    set_train_start = pyqtSignal(object, float)
    start_train = pyqtSignal(object, float)
    start_predicting = pyqtSignal(object, float)
    stop_predicting = pyqtSignal(object, float)

    def run(self):
        """ Run the LSL stream thread."""
        logging.info("Looking for an LSL stream...")
        streams = resolve_stream('type', 'Markers')

        # Create a new inlet to read from the stream
        inlet = StreamInlet(streams[0])

        while True:
            # Pull a new sample from the inlet
            marker, timestamp = inlet.pull_sample()
            timestamp = time.time()
            if (marker[0] == '98'):
                self.new_sample.emit(marker[0], timestamp)
                self.set_train_start.emit(marker[0], timestamp)
                self.new_sample.emit(marker[0], timestamp)
                date_time = datetime.fromtimestamp(timestamp)
                logging.info(f"Start of training trigger {marker[0]} written at {date_time}")
            if(marker[0] == '99'):
                self.new_sample.emit(marker[0], timestamp)
                self.start_train.emit(marker[0], timestamp)
                date_time = datetime.fromtimestamp(timestamp)
                logging.info(f"End of training trigger {marker[0]} written at {date_time}")
            elif(marker[0] == '100'):
                self.new_sample.emit(marker[0], timestamp)
                self.start_predicting.emit(marker[0], timestamp)
                date_time = datetime.fromtimestamp(timestamp)
                logging.info(f"Start application trigger {marker[0]} written at {date_time}")
            elif(marker[0] == '101'):
                self.new_sample.emit(marker[0], timestamp)
                self.stop_predicting.emit(marker[0], timestamp)
                date_time = datetime.fromtimestamp(timestamp)
                logging.info(f"Stop application trigger {marker[0]} written at {date_time}")
            else:
                # Emit the new sample data
                # logging.info(f"New sample: {marker[0]} at {timestamp}")
                self.new_sample.emit(marker[0], timestamp)

