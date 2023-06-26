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

"""Detection Trainer Pipeline"""

import argparse
import random
import warnings

from pathlib import Path
from loguru import logger as logguru
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from src.model.yolox.data import get_yolox_datadir
from src.model.yolox.core import launch
from src.model.yolox.exp import Exp as MyExp
from src.model.yolox.exp import Exp, check_exp_value
from src.model.yolox.utils import (
    configure_module,
    configure_nccl,
    configure_omp,
    get_num_devices,
)
from src.model.yolox.data import VOCDetection, TrainTransform, ValTransform
from src.model.yolox.evaluators import VOCEvaluator
from src.utils.general_utils import download_to, extract_file

# type: ignore
@logguru.catch
def main(exp: Exp, args) -> None:
    """Main function for YOLOX training"""
    assert (
        torch.cuda.is_available()
    ), "CUDA is not available. Object Detection only work on CUDA platform."

    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()  # nosec


def run_detection(cfg: DictConfig) -> None:
    """Run Detection Pipeline"""
    configure_module()
    args = argparse.Namespace(**cfg.trainer.yolox)
    if args.logger == "wandb":
        for item in cfg.model_analysis.wandb:
            args.__dict__[item] = cfg.model_analysis.wandb[item]

    url: str = cfg.data_module.dataset.url
    blob_file: str = cfg.data_module.dataset.blob_file
    root_dir: Path = Path(cfg.data_module.dataset.root_dir)

    if cfg.data_module.dataset.download:
        logguru.info(f"downloading from {url} to {blob_file} in {root_dir}")
        download_to(url, blob_file, root_dir)
        extract_file(root_dir, blob_file)

    assert cfg.data_module.dataset_format in [
        "coco",
        "voc",
    ], f"Unsupported format {cfg.data_module.dataset_format}"

    if cfg.trainer.yolox.model != "yolox_nano":
        if cfg.data_module.dataset_format == "coco":
            exp = COCO_Exp(cfg)
        if cfg.data_module.dataset_format == "voc":
            exp = VOC_Exp(cfg)
    else:
        exp = YOLOX_NANO_Exp(cfg)

    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )


class COCO_Exp(MyExp):
    def __init__(self, cfg) -> None:
        super(COCO_Exp, self).__init__()

        dataset_config = cfg.data_module.dataset
        for item in dataset_config:
            setattr(self, item, dataset_config[item])

        model_config = cfg.model[cfg.trainer.yolox.model]
        for item in model_config:
            setattr(self, item, model_config[item])

        self.seed = cfg.trainer.yolox.seed
        self.max_epoch = cfg.trainer.yolox.max_epoch
        self.output_dir = cfg.trainer.yolox.output_dir
        self.exp_name = cfg.project_name


class VOC_Exp(MyExp):
    def __init__(self, cfg) -> None:
        super(VOC_Exp, self).__init__()

        dataset_config = cfg.data_module.dataset
        for item in dataset_config:
            setattr(self, item, dataset_config[item])

        model_config = cfg.model[cfg.trainer.yolox.model]
        for item in model_config:
            setattr(self, item, model_config[item])

        self.seed = cfg.trainer.yolox.seed
        self.max_epoch = cfg.trainer.yolox.max_epoch
        self.output_dir = cfg.trainer.yolox.output_dir
        self.exp_name = cfg.project_name

    def get_dataset(self, cache: bool, cache_type: str = "ram") -> VOCDetection:
        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=self.image_sets,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs) -> VOCDetection:
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=self.image_sets,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(
        self, batch_size, is_distributed, testdev=False, legacy=False
    ) -> VOCEvaluator:
        return VOCEvaluator(
            dataloader=self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )


class YOLOX_NANO_Exp(MyExp):
    def __init__(self, cfg) -> None:
        super(YOLOX_NANO_Exp, self).__init__()

        dataset_config = cfg.data_module.dataset
        for item in dataset_config:
            setattr(self, item, dataset_config[item])

        model_config = cfg.model[cfg.trainer.yolox.model]
        for item in model_config:
            setattr(self, item, model_config[item])

        self.seed = cfg.trainer.yolox.seed
        self.max_epoch = cfg.trainer.yolox.max_epoch
        self.output_dir = cfg.trainer.yolox.output_dir
        self.exp_name = cfg.project_name

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from src.model.yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
