import sys
import os
from datetime import datetime


class Logger(object):

    def __init__(self, task=None):
        FOLDER = os.path.join(".", "LOGS")
        if task:
            folder_path = os.path.join(FOLDER, task)
        else:
            folder_path = os.path.join(FOLDER, "")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.terminal = sys.stdout
        filename = datetime.now().strftime('log_%d_%m_%Y_%H%M.log')
        filepath = os.path.join(folder_path, filename)
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def close(self):
        self.log.close()
