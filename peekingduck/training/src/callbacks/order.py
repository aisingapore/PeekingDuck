"""Definitions of Callback order."""
from enum import IntEnum


class CallbackOrder(IntEnum):
    """Callback orders."""

    HISTORY = 0
    METRICMETER = 1
    MODELCHECKPOINT = 2
    EARLYSTOPPING = 3

