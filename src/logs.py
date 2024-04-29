from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG
from time import strftime

logger = getLogger(__name__)
logger.setLevel(DEBUG)

file_handler = FileHandler(f"logs/{strftime('%Y-%m-%d_%H:%M:%S')}.log")
console_handler = StreamHandler()
formatter = Formatter("%(asctime)s %(levelname)s: %(message)s")

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Logger initialized.")
