"""Definitions of Callback order."""
from enum import IntEnum


class CallbackOrder(IntEnum):
    """Callback orders."""

    HISTORY = 0
    METRICMETER = 1
    MODELCHECKPOINT = 2
    LOGGER = 3
    EARLYSTOPPING = 4

