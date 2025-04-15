import logging
import os
import time
from brainflow import DataFilter


class DataWriter:

    def __init__(self, board_id, board, channels_layout):
        self.board_id = board_id
        self.board = board
        self.channels_layout = channels_layout

    def write_header(self, file):
        for column in self.channels_layout:
            file.write(str(column) + '\t')
        file.write('\n')

    def export_file(self, filename=None, folder='export', format='csv'):
        """
        Export the data to a file
        :param filename: str, name of the file
        :param folder: str, name of the folder
        :param format: str, format of the file
        """
        # Compose the file name using the board name and the current time
        try:
            if filename is None:
                filename = f"{self.board.get_device_name(self.board_id)}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            path = os.path.join(folder, filename + '.' + format)
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(path, 'w') as file:
                self.write_header(file)
                data = self.board.get_board_data()
                if format == 'csv':
                    DataFilter.write_file(data, path, 'a')
        except Exception as e:
            logging.error(f"Error exporting file: {e}")
