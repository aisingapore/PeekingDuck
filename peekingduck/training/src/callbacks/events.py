from enum import Enum

class EVENTS(Enum):
    """Events that are used by trainer class."""

    TRAINER_START = "on_trainer_start"
    TRAINER_END = "on_trainer_end"

    FIT_START = "on_fit_start"
    FIT_END = "on_fit_end"

    TRAIN_BATCH_START = "on_train_batch_start"
    TRAIN_BATCH_END = "on_train_batch_end"

    TRAIN_LOADER_START = "on_train_loader_start"
    TRAIN_LOADER_END = "on_train_loader_end"

    VALID_LOADER_START = "on_valid_loader_start"
    VALID_LOADER_END = "on_valid_loader_end"

    TRAIN_EPOCH_START = "on_train_epoch_start"
    TRAIN_EPOCH_END = "on_train_epoch_end"

    VALID_EPOCH_START = "on_valid_epoch_start"
    VALID_EPOCH_END = "on_valid_epoch_end"

    VALID_BATCH_START = "on_valid_batch_start"
    VALID_BATCH_END = "on_valid_batch_end"

    INFERENCE_START = "on_inference_start"
    INFERENCE_END = "on_inference_end"
