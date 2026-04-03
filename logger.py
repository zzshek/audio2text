import logging
import sys

logger = logging.getLogger("audio2text")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

_console = logging.StreamHandler(sys.stderr)
_console.setFormatter(_fmt)
logger.addHandler(_console)

_file = logging.FileHandler("audio2text.log", encoding="utf-8")
_file.setFormatter(_fmt)
logger.addHandler(_file)
