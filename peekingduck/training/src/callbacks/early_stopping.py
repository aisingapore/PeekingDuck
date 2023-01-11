"""Early Stopping Callbacks.

Logic:
    0. We keep track of the following variables:
        - best_score: the best score so far, can be loss or metric.
        - patience: the number of epochs to wait before stopping, if the score
            does not improve.
    1. This is called after each validation epoch.
    2. We check if the current epoch score is better than the best score.
    3. If it is, we change the best score to the current score and reset the patience.
    4. If it is not, we decrement the patience.
    5. If the patience is 0, we stop the training.

TODO:
    1. As usual, the issue is attributes/args such as mode is defined in `pipeline_config`,
        may need to inherit the `pipeline_config` and use it to initiate the attributes.
"""
from src.callbacks.default_callbacks import Callback
from src.trainer.default_trainer import Trainer
from src.utils.callback_utils import init_improvement


class EarlyStopping(Callback):
    """Class for Early Stopping.

    Args:
        mode (str): One of {"min", "max"}. In min mode, training will stop
            when the quantity monitored has stopped decreasing. In "max"
            mode it will stop when the quantity monitored has stopped increasing.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(
        self, mode: str, monitor: str, patience: int = 3, min_delta: float = 1e-6
    ) -> None:
        super().__init__()
        self.mode = mode
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta

    def on_trainer_start(self, trainer: Trainer) -> None:
        self.improvement, self.best_valid_score = init_improvement(
            mode=self.mode, min_delta=self.min_delta
        )
        self.patience_counter = 0
        self.stop = False
        trainer.stop = self.stop  # assign to trainer

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        valid_score = trainer.epoch_dict.get(self.monitor)
        if self.improvement(
            curr_epoch_score=valid_score, curr_best_score=self.best_valid_score
        ):
            # update self.best_valid_score
            self.best_valid_score = valid_score

            # reset patience counter
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(
                f"Early Stopping Counter {self.patience_counter} out of {self.patience}"
            )

            if self.patience_counter >= self.patience:
                self.stop = True
                trainer.stop = self.stop
                print("Early Stopping!")

    def should_stop(self):
        """The actual algorithm of early stopping.
        Consider shifting the stop logic here.
        """
