import logging
import time
import pylsl
import numpy as np
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
    previous_ts = 0

    def run(self):
        """ Run the LSL stream thread."""
        logging.info("Looking for an LSL stream...")
        streams = []

        # Create a new inlet to read from the stream, check until a stream is found
        while not streams:
            streams = resolve_stream('name', 'Cortex Markers')
            time.sleep(1)
        logging.info("LSL stream found: {}".format(streams[0].name()))
        inlet = StreamInlet(streams[0], processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_threadsafe)


        while True:
            try:
                # Pull a new sample from the inlet
                marker, timestamp = inlet.pull_sample()
                timestamp = pylsl.local_clock()
                delta_ts = np.round(timestamp - self.previous_ts, 2) if self.previous_ts != 0 else 0
                self.previous_ts = timestamp

                if (marker[0] == '98'):
                    self.new_sample.emit(marker[0], timestamp)
                    self.set_train_start.emit(marker[0], timestamp)
                    date_time = datetime.fromtimestamp(time.time())
                    logging.info(f"Start of training trigger {marker[0]} written at {date_time}")
                if(marker[0] == '99'):
                    self.new_sample.emit(marker[0], timestamp)
                    self.start_train.emit(marker[0], timestamp)
                    date_time = datetime.fromtimestamp(time.time())
                    logging.info(f"End of training trigger {marker[0]} written at {date_time}")
                elif(marker[0] == '100'):
                    self.new_sample.emit(marker[0], timestamp)
                    self.start_predicting.emit(marker[0], timestamp)
                    date_time = datetime.fromtimestamp(time.time())
                    logging.info(f"Start application trigger {marker[0]} written at {date_time}")
                elif(marker[0] == '101'):
                    self.new_sample.emit(marker[0], timestamp)
                    self.stop_predicting.emit(marker[0], timestamp)
                    date_time = datetime.fromtimestamp(time.time())
                    logging.info(f"Stop application trigger {marker[0]} written at {date_time}")
                else:
                    # Emit the new sample data
                    logging.debug(f"New sample: {marker[0]} after {delta_ts}")
                    self.new_sample.emit(marker[0], timestamp)
            except Exception as e:
                logging.error(f"Error while reading LSL stream: {e}")
                streams = resolve_stream('name', 'Cortex Markers')
                logging.info("LSL stream found: {}".format(streams[0].name()))
                inlet = StreamInlet(streams[0],
                                    processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_threadsafe)

