"""Information-theoretic metrics on game pairs."""

from conway_foundations.synergy.metrics import METRIC_REGISTRY

from . import metrics_ei as _metrics_ei  # noqa: F401
from . import metrics_emergence as _metrics_emergence  # noqa: F401
from . import metrics_kl as _metrics_kl  # noqa: F401
from . import metrics_markov as _metrics_markov  # noqa: F401
from . import metrics_phi as _metrics_phi  # noqa: F401
from . import metrics_te as _metrics_te  # noqa: F401

__all__ = ["METRIC_REGISTRY"]
