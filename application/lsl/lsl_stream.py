import json
import logging
import time
import pylsl
import numpy as np
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from pylsl import resolve_stream, StreamInlet, StreamOutlet
from processing.proc_helper import freq_bands
from utils.net_utils import convert_to_serializable


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
        inlet = StreamInlet(streams[0],
                            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_threadsafe)

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
                if (marker[0] == '99'):
                    self.new_sample.emit(marker[0], timestamp)
                    self.start_train.emit(marker[0], timestamp)
                    date_time = datetime.fromtimestamp(time.time())
                    logging.info(f"End of training trigger {marker[0]} written at {date_time}")
                elif (marker[0] == '100'):
                    self.new_sample.emit(marker[0], timestamp)
                    self.start_predicting.emit(marker[0], timestamp)
                    date_time = datetime.fromtimestamp(time.time())
                    logging.info(f"Start application trigger {marker[0]} written at {date_time}")
                elif (marker[0] == '101'):
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


def start_lsl_eeg_stream(channels, fs, source_id, stream_name='Cortex EEG', type='EEG'):
    """
    Start an LSL stream for the EEG data

    :param channels: list of str, channel names
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        ch_count = len(channels) + 1
        info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
                                nominal_srate=fs, channel_format='float32',
                                source_id=source_id)
        # Add channel names
        chs = info.desc().append_child("channels")
        for ch in channels:
            chs.append_child("channel").append_child_value("name", ch)
        chs.append_child("channel").append_child_value("name", "Trigger")
        eeg_outlet = StreamOutlet(info)
        logging.info(f"LSL EEG stream started {info.name()}")
        return eeg_outlet
    except Exception as e:
        logging.error(f"Error starting LSL stream: {e}")


def start_lsl_power_bands_stream(channels, fs, source_id, stream_name='Cortex PSD', type='PSD'):
    """
    Start an LSL stream for the power bands data
    :param channels: list of str, channel names
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        ch_count = len(channels)
        info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
                                nominal_srate=fs, channel_format='float32',
                                source_id=source_id)
        # Add channel names
        chs = info.desc().append_child("channels")
        for ch in channels:
            chs.append_child("channel").append_child_value("name", ch)
        band_powers_outlet = StreamOutlet(info)
        logging.info(f"LSL power bands stream started {info.name()}")
        return band_powers_outlet
    except Exception as e:
        logging.error(f"Error starting LSL stream: {e}")


def start_lsl_prediction_stream(fs, source_id, stream_name='Cortex Inference', type='Markers'):
    """
    Start an LSL stream for the prediction data
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return: StreamOutlet object
    """
    try:
        info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=1,
                                nominal_srate=fs, channel_format='string',
                                source_id=source_id)
        prediction_outlet = pylsl.StreamOutlet(info)
        logging.info(f"LSL prediction stream started {info.name()}")
        return prediction_outlet
    except Exception as e:
        logging.error(f"Error starting LSL stream: {e}")


def start_lsl_quality_stream(channels, fs, source_id, stream_name='Cortex Qualities', type='Qualities'):
    """ Start an LSL stream for the quality dat
    :param channels: list of str, channel names
    :param fs: int, sampling rate
    :param source_id: str, source id
    :param stream_name: str, name of the LSL stream
    :param type: str, type of the LSL stream
    :return StreamOutlet object
    """
    try:
        ch_count = len(channels)
        info = pylsl.StreamInfo(name=stream_name, type=type, channel_count=ch_count,
                                nominal_srate=fs, channel_format='float32',
                                source_id=source_id)
        quality_outlet = StreamOutlet(info)
        logging.info(f"LSL quality stream started {info.name()}")
        return quality_outlet
    except Exception as e:
        logging.error(f"Error starting LSL stream: {e}")


def push_lsl_raw_eeg(outlet: StreamOutlet, data, start_eeg, end_eeg, counter, ts=0, chunk=False):
    """
    Push a chunk of data to the LSL stream
    :param outlet: StreamOutlet object
    :param data: numpy array of shape (n_channels, n_samples)
    :param start_eeg: int, start index of the EEG channels
    :param end_eeg: int, end index of the EEG channels
    :param counter: int, chunk counter
    :param ts: float, timestamp value
    :param chunk: bool, whether to push a chunk of data or a single sample
    """
    try:
        # Get EEG and Trigger from data and push it to LSL
        eeg = data[start_eeg:end_eeg]
        trigger = data[-1]

        # Horizontal stack EEG and Trigger
        eeg = np.concatenate((eeg, trigger.reshape(1, len(trigger))), axis=0)

        ts_to_lsl_offset = time.time() - pylsl.local_clock()
        # Get only the seconds part of the timestamp
        ts = ts - ts_to_lsl_offset
        if chunk:
            outlet.push_chunk(eeg.T.tolist(), ts)
            logging.debug(f"Pushed chunk {counter} to LSL stream {outlet.get_info().name()}")
        else:
            for i in range(eeg.shape[1]):
                sample = eeg[:, i]
                outlet.push_sample(sample.tolist(), ts[i])
            logging.debug(f"Pushed {eeg.shape[1]} samples  of chunk {counter} to LSL stream {outlet.get_info().name()}")
    except Exception as e:
        logging.error(f"Error pushing chunk to LSL: {e}")


def push_lsl_band_powers(outlet: StreamOutlet, band_powers, timestamp):
    """
    Push the power bands to the LSL stream
    :param outlet: StreamOutlet object
    :param band_powers: list of band power values
    :param timestamp: float, timestamp value
    """
    try:
        outlet.push_chunk(band_powers.tolist(), timestamp)
        logging.debug(
            f"Pushed band powers {' '.join(list(freq_bands.keys()))} to LSL stream {outlet.get_info().name()}")
    except Exception as e:
        logging.error(f"Error pushing band powers to LSL: {e}")


def push_lsl_prediction(outlet: StreamOutlet, prediction):
    """
    Push a prediction to the LSL stream
    :param outlet: StreamOutlet object
    :param prediction: dict, prediction data
    """
    try:
        # Serialize the dictionary to a JSON string
        prediction_json = json.dumps(prediction, default=convert_to_serializable)
        outlet.push_sample([prediction_json])
        logging.debug(f"Pushed prediction {prediction} to LSL stream {outlet.get_info().name()} ")
    except Exception as e:
        logging.error(f"Error pushing prediction to LSL: {e}")


def push_lsl_quality(outlet: StreamOutlet, quality):
    """
    Push a quality indicator to the LSL stream
    :param outlet: StreamOutlet object
    :param quality: list of quality indicators
    """
    try:
        outlet.push_sample(quality)
        logging.debug(f"Pushed quality {quality} to LSL stream {outlet.get_info().name()}")
    except Exception as e:
        logging.error(f"Error pushing quality to LSL: {e}")
