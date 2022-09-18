class Error(Exception):
    def __init__(self, message, exitcode=1):
        self.message = message
        self.exitcode = exitcode
