from importlib.metadata import PackageNotFoundError, version

from docdistance import config  # noqa: F401  (sets up logging + paths on import)
from docdistance.distance import (
    DistanceResult,
    SourceConditionedResult,
    closeness,
    compute_distance,
    compute_source_conditioned,
    rwmd,
    smd,
    wcd,
)
from docdistance.pipeline import (
    DocDistance,
    document_distance,
    source_conditioned_distance,
)

try:
    __version__ = version("docdistance")
except PackageNotFoundError:  # running from source, not installed
    __version__ = "0.0.0"

__all__ = [
    "DocDistance",
    "DistanceResult",
    "SourceConditionedResult",
    "document_distance",
    "source_conditioned_distance",
    "compute_distance",
    "compute_source_conditioned",
    "smd",
    "wcd",
    "rwmd",
    "closeness",
    "__version__",
]
