from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

########### SETUP ###############

# set up logger - INFO by default (DEBUG only via the CLI --verbose flag), sink to stderr so
# stdout stays clean for --json / --result-only output
logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end="", file=sys.stderr), colorize=True, level="INFO")
except ModuleNotFoundError:
    pass

########## VARIABLES ############

# Load environment variables from .env file if it exists
load_dotenv()

# paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# log current root dir (debug so it never pollutes machine-readable stdout)
logger.debug(f"PROJ_ROOT path is: {PROJ_ROOT}")

########## MODELS ###############

# segmenter (wtpsplit SaT) and the mmBERT statement encoders, by backend
SAT_MODEL = "sat-3l-sm"
MMBERT_TORCH_MODEL = "jhu-clsp/mmBERT-base"
MMBERT_OPENVINO_LOCAL = MODELS_DIR / "02-mmbert-openvino-int8"
MMBERT_OPENVINO_HF = "stellars/mmBERT-base-openvino-int8"


def configure_logging(verbose: bool = False) -> None:
    """Re-point loguru at stderr at INFO, or DEBUG when ``verbose`` - the CLI calls this first.

    stderr keeps stdout reserved for the result so ``--json`` and ``--result-only`` stay machine-parseable.
    """
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    try:
        from tqdm import tqdm

        logger.add(
            lambda msg: tqdm.write(msg, end="", file=sys.stderr), colorize=True, level=level
        )
    except ModuleNotFoundError:
        logger.add(sys.stderr, colorize=True, level=level)
