import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# StreamHandler that flushes every message (required for DVC / piped output)
_stream_handler = logging.StreamHandler()
_stream_handler.flush = lambda: _stream_handler.stream.flush()

logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        _stream_handler,
    ],
)