"""
Loss functions Class to interact with Tensorflow

    @staticmethod
    def CategoricalCrossentropy(parameters):
        return tf.keras.losses.CategoricalCrossentropy(**parameters)

    @staticmethod
    def SparseCategoricalCrossentropy(parameters):
        return tf.keras.losses.SparseCategoricalCrossentropy(**parameters)

"""

from typing import Any, Dict
from omegaconf import DictConfig
import tensorflow as tf
import torch

class LossAdapter:

    @staticmethod
    def get_tensorflow_loss_func(name, parameters: DictConfig = {}):
        return getattr(tf.keras.losses, name)(**parameters) if len(parameters) > 0 else getattr(tf.keras.losses, name)()

    @staticmethod
    def compute_criterion( # for pytorch trainer
        y_trues: torch.Tensor,
        y_logits: torch.Tensor,
        criterion_params: Dict[str, Any],
        stage: str,
    ) -> torch.Tensor:
        """Train Loss Function.
        Note that we can evaluate train and validation fold with different loss functions.
        The below example applies for CrossEntropyLoss.
        Args:
            y_trues ([type]): Input - N,C) where N = number of samples and C = number of classes.
            y_logits ([type]): If containing class indices, shape (N) where each value is
                $0 \leq \text{targets}[i] \leq C-10≤targets[i]≤C-1$.
                If containing class probabilities, same shape as the input.
            stage (str): train or valid, sometimes people use different loss functions for
                train and valid.
        """

        if stage == "train":
            loss_fn = getattr(torch.nn, criterion_params.train_criterion)(
                **criterion_params.train_criterion_params
            )
        elif stage == "valid":
            loss_fn = getattr(torch.nn, criterion_params.valid_criterion)(
                **criterion_params.valid_criterion_params
            )
        loss = loss_fn(y_logits, y_trues)
        return loss

    @staticmethod
    def get_lr(optimizer: torch.optim) -> float: # for pytorch trainer
        """Get the learning rate of optimizer for the current epoch.
        Note learning rate can be different for different layers, hence the for loop.
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]

