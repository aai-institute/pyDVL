"""
WARNING:
    This is mostly just a quick hack for testing and debugging
"""
import logging
import logging.handlers
import pickle
import socketserver
import struct
from multiprocessing import Process
from typing import Optional


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.

    From the Logging Cookbook:
    https://docs.python.org/3.8/howto/logging-cookbook.html
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack(">L", chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = pickle.loads(chunk)
            record = logging.makeLogRecord(obj)
            self.handle_log_record(record)

    def handle_log_record(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger("server")
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """Simple TCP socket-based logging receiver suitable for testing.
    Almost verbatim from the Logging Cookbook:
    https://docs.python.org/3.8/howto/logging-cookbook.html
    """

    allow_reuse_address = True

    def __init__(self, host: str, port: int, handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        import select

        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort


def start_logging_server(
    host: str = "localhost", port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT
) -> Process:
    global server
    import platform

    if server is not None:
        return server
    if platform.system() != "Linux":
        from multiprocessing import set_start_method

        set_start_method("fork")

    logging.basicConfig(
        format="%(relativeCreated)5d %(name)-15s %(levelname)-8s %(message)s"
    )
    tcpserver = LogRecordSocketReceiver(host, port)
    server = Process(target=tcpserver.serve_until_stopped, daemon=True)
    server.start()
    return server


def set_logger(
    host: str = "localhost",
    port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    _logger=None,
):
    global logger
    if _logger is not None:
        logger = _logger
    elif logger is None:
        import logging.handlers

        logger = logging.getLogger("root")
        logger.setLevel(logging.DEBUG)
        # socket handler sends the raw event, pickled
        socketHandler = logging.handlers.SocketHandler(host, port)
        logger.addHandler(socketHandler)


server: Optional[Process] = None
logger = None

set_logger()
