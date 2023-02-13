# Copyright 2023 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import torchmetrics
from torchmetrics.classification.stat_scores import (
    MulticlassStatScores,
)  # for type hinting, referenced from PyTorch Lightning source code
import pandas as pd
import tabulate
import torch
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassCalibrationError
from typing import List, Dict
from omegaconf import DictConfig
from src.metrics.base import MetricsAdapter


class PytorchMetrics(MetricsAdapter):
    @classmethod
    def get_metrics(cls, task, num_classes, metric_list: list) -> MetricCollection:
        metric_collection_list = []

        for metric in metric_list:
            metric_collection_list.append(cls.get_metric(task, num_classes, metric))
        metrics_collection: MetricCollection = MetricCollection(metric_collection_list)
        return metrics_collection

    @classmethod
    def get_metric(
        cls, task: str, num_classes: int, metric
    ) -> MulticlassStatScores:  # the metric can be a dict or list
        """
        Refer to TorchMetrics implementation
        """
        if type(metric) is str:
            torch_metric = getattr(torchmetrics, metric)(
                num_classes=num_classes, task=task
            )

        elif type(metric) is DictConfig:
            for metric_name, metric_params in metric.items():
                torch_metric = getattr(torchmetrics, metric_name)(
                    num_classes=num_classes, task=task, **metric_params
                )

        else:
            raise TypeError

        return torch_metric

    @staticmethod
    def get_classification_metrics(
        metrics,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
    ):
        """[summary]
        # https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);
            mode (str, optional): [description]. Defaults to "valid".
        """

        train_metrics = metrics.clone(prefix="train_")
        valid_metrics = metrics.clone(prefix="val_")

        # FIXME: currently train and valid give same results, since this func call takes in
        # y_trues, etc from valid_one_epoch.
        train_metrics_results = train_metrics(y_probs, y_trues.flatten())
        train_metrics_results_df = pd.DataFrame.from_dict([train_metrics_results])

        valid_metrics_results = valid_metrics(y_probs, y_trues.flatten())
        valid_metrics_results_df = pd.DataFrame.from_dict([valid_metrics_results])

        # TODO: relinquish this logging duty to a callback or for now in train_one_epoch and valid_one_epoch.
        # self.logger.info(
        #     f"\ntrain_metrics:\n{tabulate(train_metrics_results_df, headers='keys', tablefmt='psql')}\n"
        # )
        # self.logger.info(
        #     f'\nvalid_metrics:\n{tabulate(valid_metrics_results_df, headers="keys", tablefmt="psql")}\n'
        # )

        return train_metrics_results, valid_metrics_results

    @staticmethod
    def get_classification_metrics(
        metrics: MetricCollection,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
    ):
        """[summary]
        # https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);
            mode (str, optional): [description]. Defaults to "valid".
        """

        train_metrics = metrics.clone(prefix="train_")
        valid_metrics = metrics.clone(prefix="val_")

        # FIXME: currently train and valid give same results, since this func call takes in
        # y_trues, etc from valid_one_epoch.
        train_metrics_results = train_metrics(y_probs, y_trues.flatten())
        train_metrics_results_df = pd.DataFrame.from_dict([train_metrics_results])

        valid_metrics_results = valid_metrics(y_probs, y_trues.flatten())
        valid_metrics_results_df = pd.DataFrame.from_dict([valid_metrics_results])

        return train_metrics_results, train_metrics_results_df, valid_metrics_results, valid_metrics_results_df
