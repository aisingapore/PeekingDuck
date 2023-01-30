from enum import Enum

class EVENTS(Enum):
    """Events that are used by trainer class."""

    ON_TRAINER_START = "on_trainer_start"
    ON_TRAINER_END = "on_trainer_end"

    ON_FIT_START = "on_fit_start"
    ON_FIT_END = "on_fit_end"

    ON_TRAIN_BATCH_START = "on_train_batch_start"
    ON_TRAIN_BATCH_END = "on_train_batch_end"

    ON_TRAIN_LOADER_START = "on_train_loader_start"
    ON_TRAIN_LOADER_END = "on_train_loader_end"

    ON_VALID_LOADER_START = "on_valid_loader_start"
    ON_VALID_LOADER_END = "on_valid_loader_end"

    ON_TRAIN_EPOCH_START = "on_train_epoch_start"
    ON_TRAIN_EPOCH_END = "on_train_epoch_end"

    ON_VALID_EPOCH_START = "on_valid_epoch_start"
    ON_VALID_EPOCH_END = "on_valid_epoch_end"

    ON_VALID_BATCH_START = "on_valid_batch_start"
    ON_VALID_BATCH_END = "on_valid_batch_end"

    ON_INFERENCE_START = "on_inference_start"
    ON_INFERENCE_END = "on_inference_end"
