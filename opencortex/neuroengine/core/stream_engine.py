"""
StreamEngine - Main application controller and independent service
Runs as the primary loop, with GUI as an optional interface.

Author: Michele Romani
"""

import threading
import time
import numpy as np
import logging
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from brainflow.board_shim import BoardShim, BoardIds

from opencortex.neuroengine.classifier import Classifier
from opencortex.neuroengine.flux.base import Parallel
from opencortex.neuroengine.flux.band_power import BandPowerExtractor
from opencortex.neuroengine.flux.quality_estimator import QualityEstimator
from opencortex.neuroengine.network.lsl_stream import (
    start_lsl_eeg_stream, start_lsl_power_bands_stream,
    start_lsl_inference_stream, start_lsl_quality_stream,
    push_lsl_raw_eeg, push_lsl_band_powers, push_lsl_inference, push_lsl_quality
)
from opencortex.utils.layouts import layouts


@dataclass
class StreamData:
    """Data packet sent to interfaces"""
    raw_eeg: np.ndarray
    filtered_eeg: np.ndarray
    band_powers: np.ndarray
    quality_scores: list
    timestamp: float
    trigger: int


@dataclass
class Command:
    """Command sent to StreamEngine"""
    action: str  # 'set_inference_mode', 'train', 'send_trigger', etc.
    params: Dict[str, Any]
    callback: Optional[Callable] = None


class StreamEngine:
    """
    Main application controller that runs independently.
    Can operate headless or with GUI attached.
    """

    def __init__(self, board, config, window_size=1):
        self.board = board
        self.config = config
        self.window_size = window_size

        # Core properties
        self.board_id = self.board.get_board_id()
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.num_points = self.window_size * self.sampling_rate

        # Engine state
        self.running = False
        self.inference_mode = False
        self.first_prediction = True

        # Configuration
        self.model = config.get('model', 'LDA')
        self.proba = config.get('proba', False)
        self.group_predictions = config.get('group_predictions', False)
        self.nclasses = config.get('nclasses', 3)
        self.flash_time = config.get('flash_time', 250)
        self.epoch_length_ms = config.get('epoch_length_ms', 1000)
        self.baseline_ms = config.get('baseline_ms', 100)
        self.quality_thresholds = config.get('quality_thresholds',
            [(-100, -50, 'yellow', 0.5), (-50, 50, 'green', 1.0), (50, 100, 'yellow', 0.5)])
        self.over_sample = config.get('oversample', True)
        self.update_interval_ms = config.get('update_buffer_speed_ms', 100)

        # Processing pipeline
        self.pipeline = Parallel(
            band_power=BandPowerExtractor(fs=self.sampling_rate, ch_names=self.eeg_channels),
            quality=QualityEstimator(quality_thresholds=self.quality_thresholds)
        )

        # Data buffers
        self.filtered_eeg = np.zeros((len(self.eeg_channels) + 1, self.num_points))
        self.raw_data = None

        # Threading and communication
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.command_queue = queue.Queue()
        self.main_thread = None
        self.classifier = None

        # Interface callbacks (GUI, API, etc.)
        self.data_callbacks = []
        self.event_callbacks = []

        # Timing calculations
        self._calculate_timing_parameters()

        # Initialize LSL outlets
        self._init_lsl_outlets()

        logging.info("StreamEngine initialized")

    def _calculate_timing_parameters(self):
        """Calculate timing parameters for predictions and epochs."""
        self.off_time = (self.flash_time * (self.nclasses - 1))
        self.prediction_interval = int(2 * self.flash_time + self.off_time)
        self.epoch_data_points = int(self.epoch_length_ms * self.sampling_rate / 1000)
        self.inference_ms = self.baseline_ms + (self.flash_time * self.nclasses) + self.epoch_length_ms
        self.prediction_datapoints = int(self.inference_ms * self.sampling_rate / 1000)

        self.slicing_trigger = (self.epoch_length_ms + self.baseline_ms) // self.flash_time
        if self.slicing_trigger > self.nclasses:
            self.slicing_trigger = self.nclasses

    def _init_lsl_outlets(self):
        """Initialize LSL output streams."""
        device_name = self.board.get_device_name(self.board_id)
        eeg_names = BoardShim.get_eeg_names(self.board_id)

        self.eeg_outlet = start_lsl_eeg_stream(
            channels=eeg_names, fs=self.sampling_rate, source_id=device_name)
        self.inference_outlet = start_lsl_inference_stream(
            channels=1, fs=self.sampling_rate, source_id=device_name)
        self.band_powers_outlet = start_lsl_power_bands_stream(
            channels=eeg_names, fs=self.sampling_rate, source_id=device_name)
        self.quality_outlet = start_lsl_quality_stream(
            channels=eeg_names, fs=self.sampling_rate, source_id=device_name)

    # ===================== MAIN ENGINE CONTROL =====================

    def start(self):
        """Start the StreamEngine main loop."""
        if self.running:
            logging.warning("StreamEngine already running")
            return

        self.running = True

        # Initialize classifier in background
        if self.model is not None:
            self.executor.submit(self._init_classifier)

        # Start main processing loop
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()

        logging.info("StreamEngine started")

    def stop(self):
        """Stop the StreamEngine."""
        self.running = False

        if self.main_thread:
            self.main_thread.join(timeout=5.0)

        self.executor.shutdown(wait=True)
        logging.info("StreamEngine stopped")

    def _main_loop(self):
        """Main processing loop - runs independently."""
        last_update = time.time()

        while self.running:
            current_time = time.time()

            # Process commands from interfaces
            self._process_commands()

            # Update data at configured interval
            if (current_time - last_update) * 1000 >= self.update_interval_ms:
                try:
                    self._update_data()
                    last_update = current_time
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    self._notify_event('error', {'message': str(e)})

            # Small sleep to prevent 100% CPU usage
            time.sleep(0.001)

    def _update_data(self):
        """Core data processing - heart of the engine."""
        try:
            # Get raw data from board
            data = self.board.get_current_board_data(num_samples=self.num_points)
            self.raw_data = data

            # Extract and filter EEG
            start_eeg = layouts[self.board_id]["eeg_start"]
            end_eeg = layouts[self.board_id]["eeg_end"]
            eeg = data[start_eeg:end_eeg]

            # Update filtered buffer
            for count, channel in enumerate(self.eeg_channels):
                self.filtered_eeg[count] = eeg[count]

            # Extract trigger and timestamp
            trigger = data[-1]
            ts_channel = self.board.get_timestamp_channel(self.board_id)
            ts = data[ts_channel]
            self.filtered_eeg[-1] = trigger

            # Process through pipeline
            outputs = self.pipeline(self.filtered_eeg[0:len(self.eeg_channels)])
            band_powers = outputs["band_power"]
            quality_scores = outputs["quality"]

            # Push to LSL streams
            push_lsl_band_powers(self.band_powers_outlet, band_powers.to_numpy(), ts)
            push_lsl_quality(self.quality_outlet, quality_scores)
            push_lsl_raw_eeg(self.eeg_outlet, self.filtered_eeg, start_eeg, end_eeg, 0, ts, True)  # chunk_data=True

            # Create data packet for interfaces
            stream_data = StreamData(raw_eeg=self.raw_data.copy(),
                                     filtered_eeg=self.filtered_eeg.copy(),
                                     band_powers=band_powers.to_numpy(),
                                     quality_scores=quality_scores,
                                     timestamp=time.time(),
                                     trigger=int(trigger[-1]) if len(trigger) > 0 else 0)
            # Notify all registered interfaces
            self._notify_data_update(stream_data)

        except Exception as e:
            logging.error(f"Error updating data: {e}")
            raise

    # ===================== INTERFACE MANAGEMENT =====================

    def register_data_callback(self, callback: Callable[[StreamData], None]):
        """Register a callback for data updates (GUI, API, etc.)"""
        self.data_callbacks.append(callback)
        logging.info(f"Registered data callback: {callback.__name__}")

    def register_event_callback(self, callback: Callable[[str, Dict], None]):
        """Register a callback for events (predictions, errors, etc.)"""
        self.event_callbacks.append(callback)
        logging.info(f"Registered event callback: {callback.__name__}")

    def unregister_data_callback(self, callback):
        """Unregister a data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)

    def unregister_event_callback(self, callback):
        """Unregister an event callback"""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)

    def _notify_data_update(self, data: StreamData):
        """Notify all interfaces of data update."""
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Error in data callback {callback.__name__}: {e}")

    def _notify_event(self, event_type: str, data: Dict[str, Any]):
        """Notify all interfaces of an event."""
        for callback in self.event_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logging.error(f"Error in event callback {callback.__name__}: {e}")

    # ===================== COMMAND PROCESSING =====================

    def send_command(self, action: str, params: Dict[str, Any] = None, callback: Callable = None):
        """Send a command to the StreamEngine (thread-safe)."""
        command = Command(action=action, params=params or {}, callback=callback)
        self.command_queue.put(command)

    def _process_commands(self):
        """Process pending commands from the queue."""
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                self._execute_command(command)
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Error processing command: {e}")

    def _execute_command(self, command: Command):
        """Execute a single command."""
        action = command.action
        params = command.params

        try:
            if action == 'set_inference_mode':
                self._set_inference_mode(params.get('mode'))
            elif action == 'send_trigger':
                self._send_trigger(params.get('trigger', 1), params.get('timestamp', 0))
            elif action == 'train_classifier':
                self._train_classifier(params.get('data'))
            elif action == 'predict':
                self._predict_class()
            elif action == 'configure_filters':
                self._configure_filters(params)
            else:
                logging.warning(f"Unknown command: {action}")

            # Call callback if provided
            if command.callback:
                command.callback(True, None)

        except Exception as e:
            logging.error(f"Error executing command {action}: {e}")
            if command.callback:
                command.callback(False, str(e))

    # ===================== CORE FUNCTIONALITY =====================

    def _init_classifier(self):
        """Initialize the classifier."""
        try:
            self.classifier = Classifier(model=self.model, board_id=self.board_id)
            self._notify_event('classifier_ready', {})
            logging.info(f"Classifier {self.model} initialized")
        except Exception as e:
            logging.error(f"Error initializing classifier: {e}")
            self._notify_event('error', {'message': f"Classifier init failed: {e}"})

    def _set_inference_mode(self, mode=None):
        """Set inference mode."""
        if mode is None:
            self.inference_mode = not self.inference_mode
        else:
            self.inference_mode = mode

        if self.classifier:
            self.classifier.set_inference_mode(self.inference_mode)

        self._notify_event('inference_mode_changed', {'mode': self.inference_mode})
        logging.info(f"Inference mode: {'ON' if self.inference_mode else 'OFF'}")

    def _send_trigger(self, trigger=1, timestamp=0):
        """Send a trigger."""
        if timestamp == 0:
            timestamp = time.time()

        self.board.insert_marker(int(trigger))
        self._notify_event('trigger_sent', {'trigger': trigger, 'timestamp': timestamp})

        # Handle prediction triggers
        if self.inference_mode:
            if int(trigger) == self.slicing_trigger and not self.first_prediction:
                self._predict_class()
            elif int(trigger) == self.slicing_trigger and self.first_prediction:
                logging.debug('Skipping first prediction')
                self.first_prediction = False

    def _train_classifier(self, data=None):
        """Train the classifier."""
        if self.classifier is None:
            self._notify_event('error', {'message': 'Classifier not initialized'})
            return

        # Train in background
        self.executor.submit(self._train_classifier_async, data)

    def _train_classifier_async(self, data):
        """Train classifier asynchronously."""
        try:
            if data is None:
                # Get training data from current buffer
                data = self.raw_data

            self.classifier.train(data, oversample=self.over_sample)
            self._notify_event('training_complete', {})
            logging.info("Training completed")
        except Exception as e:
            logging.error(f"Training error: {e}")
            self._notify_event('error', {'message': f"Training failed: {e}"})

    def _predict_class(self):
        """Predict class."""
        if self.classifier is None:
            return

        try:
            inference_sample = self.board.get_current_board_data(self.prediction_datapoints)
            self.executor.submit(self._predict_class_async, inference_sample)
        except Exception as e:
            logging.error(f"Prediction error: {e}")

    def _predict_class_async(self, data):
        """Predict class asynchronously."""
        try:
            output = self.classifier.predict(data, proba=self.proba, group=self.group_predictions)
            push_lsl_inference(self.inference_outlet, output)
            self._notify_event('prediction_ready', {'prediction': output})
            logging.info(f"Predicted: {output}")
        except Exception as e:
            logging.error(f"Prediction error: {e}")

    def _configure_filters(self, filter_config):
        """Configure data filters."""
        # This will be implemented when we move filter logic
        pass

    # ===================== UTILITY METHODS =====================

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'running': self.running,
            'inference_mode': self.inference_mode,
            'classifier_ready': self.classifier is not None,
            'board_id': self.board_id,
            'sampling_rate': self.sampling_rate,
            'eeg_channels': len(self.eeg_channels)
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()


class HeadlessStreamEngine(StreamEngine):
    """
    Headless version that can run without any GUI.
    Perfect for server deployments, background processing, etc.
    """

    def __init__(self, board, config, window_size=1, log_file=None):
        super().__init__(board, config, window_size)

        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def run_forever(self):
        """Run the engine indefinitely (for server mode)."""
        self.start()
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Received interrupt signal")
        finally:
            self.stop()


# ===================== EXAMPLE USAGE =====================

def example_headless_usage():
    """Example of running StreamEngine headlessly."""
    from brainflow.board_shim import BoardShim, BrainFlowInputParams

    # Initialize board
    params = BrainFlowInputParams()

    board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)  # Synthetic board for testing

    # Load config
    config = {
        'model': 'LDA',
        'nclasses': 3,
        'flash_time': 250,
        'update_buffer_speed_ms': 100
    }

    # Create headless engine
    engine = HeadlessStreamEngine(board, config)

    # Register callbacks for monitoring
    def data_monitor(data: StreamData):
        print(f"Data update: {data.timestamp}")

    def event_monitor(event_type: str, data: Dict):
        print(f"Event: {event_type} - {data}")

    engine.register_data_callback(data_monitor)
    engine.register_event_callback(event_monitor)

    # Start streaming
    board.prepare_session()
    board.start_stream()

    # Run engine
    engine.run_forever()

    # Cleanup
    board.stop_stream()
    board.release_session()


if __name__ == "__main__":
    # Run headless example
    example_headless_usage()