import sys
from datetime import datetime


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        filename = datetime.now().strftime('log_%d_%m_%Y_%H%M.log')
        self.log = open(filename, "a")

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