import logging

from PyQt5.QtCore import QThread, pyqtSignal
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
import socket


class OscStreamThread(QThread):
    message_received = pyqtSignal(str, list)

    def __init__(self, listen_host='127.0.0.1', listen_port=8000,
                 send_host='127.0.0.1', send_port=9000):
        super().__init__()
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.send_host = send_host
        self.send_port = send_port

        self._running = False
        self.server = None
        self.client = None

    def _default_handler(self, address, *args):
        self.message_received.emit(address, list(args))
        logging.info(f"Received message: {address} with args: {args}")

    def run(self):
        # Setup dispatcher
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._default_handler)

        # Setup OSC server
        try:
            self.server = BlockingOSCUDPServer((self.listen_host, self.listen_port), dispatcher)
            self.client = SimpleUDPClient(self.send_host, self.send_port)
            self._running = True
            logging.info(f"OSC server listening on {self.listen_host}:{self.listen_port}")
            logging.info(f"OSC client sending to {self.send_host}:{self.send_port}")

            while self._running:
                self.server.handle_request()  # Blocking call for one message
        except socket.error as e:
            logging.error(f"Socket error: {e}")
        finally:
            logging.info("OSC server stopped")

    def send_message(self, address: str, args: list):
        if self.client:
            logging.info(f"Sending message to {address}")
            self.client.send_message(address, args)

    def stop(self):
        self._running = False
        if self.server:
            self.server.server_close()
            self.server = None
        self.client = None
