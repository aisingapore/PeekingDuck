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

import os
import argparse
import random
import warnings
import torch
import torch.backends.cudnn as cudnn

from loguru import logger as logguru
from omegaconf import DictConfig
from src.model.yolox.data import get_yolox_datadir
from src.model.yolox.core import launch
from src.model.yolox.exp import Exp as MyExp
from src.model.yolox.exp import Exp, check_exp_value
from src.model.yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


@logguru.catch
def main(exp: Exp, args):
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
    trainer.train()

def run_detection(cfg: DictConfig):
    configure_module()
    args = argparse.Namespace(**cfg)
    
    assert cfg.trainer_params.ds_format in [
        "coco",
        "voc",
    ], f"Unsupported format {cfg.trainer_params.ds_format}"

    if cfg.trainer_params.ds_format == "coco":
        exp = COCO_Exp(cfg.trainer_params.coco)
    if cfg.trainer_params.ds_format == "voc":
        exp = VOC_Exp(cfg.trainer_params.voc)
    
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
    def __init__(self, cfg):
        super(COCO_Exp, self).__init__()
        for item in cfg:
            setattr(self, item, cfg[item])


class VOC_Exp(MyExp):
    def __init__(self, cfg):
        super(VOC_Exp, self).__init__()
        for item in cfg:
            setattr(self, item, cfg[item])

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from src.model.yolox.data import VOCDetection, TrainTransform

        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=self.image_sets,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from src.model.yolox.data import VOCDetection, ValTransform
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=self.image_sets,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from src.model.yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
