# Copyright 2022 AI Singapore
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

"""Module to convert PyTorch YOLOX models to ONNX"""

import logging
from time import perf_counter

import numpy as np
import onnx
import torch
import torch.onnx
from torch import nn

from peekingduck.nodes.model.yoloxv1.yolox_files.model import YOLOX

####################
# Globals
####################
MODEL_WEIGHTS_DIR = "../../peekingduck_weights"
MODEL_MAP = {
    "yolox-tiny": {
        "data_shape": (1, 416, 416, 3),
        "path": MODEL_WEIGHTS_DIR + "/yolox/pytorch/yolox-tiny.pth",
        "size": {"depth": 0.33, "width": 0.375},
    },
    "yolox-s": {
        "data_shape": (1, 416, 416, 3),
        "path": MODEL_WEIGHTS_DIR + "/yolox/pytorch/yolox-s.pth",
        "size": {"depth": 0.33, "width": 0.5},
    },
    "yolox-m": {
        "data_shape": (1, 416, 416, 3),
        "path": MODEL_WEIGHTS_DIR + "/yolox/pytorch/yolox-m.pth",
        "size": {"depth": 0.67, "width": 0.75},
    },
    "yolox-l": {
        "data_shape": (1, 416, 416, 3),
        "path": MODEL_WEIGHTS_DIR + "/yolox/pytorch/yolox-l.pth",
        "size": {"depth": 1.0, "width": 1.0},
    },
}
YOLOX_DIR = MODEL_WEIGHTS_DIR + "/yolox"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_model(model_code: str) -> None:
    """Convert model given by 'model_code' from PyTorch to Onnx format.

    Args:
        model_code (str): supported codes
                          { "yolox-m", "yolox-l", "yolox-s", "yolox-tiny" }
    """
    logger.info(f"Convert {model_code} to Onnx")
    model_path = MODEL_MAP[model_code]["path"]
    model_size = MODEL_MAP[model_code]["size"]
    onnx_model_save_path = f"{YOLOX_DIR}/{model_code}.onnx"
    model = YOLOX(80, model_size["depth"], model_size["width"])
    model.eval()

    logger.info(f"Loading {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.head.decode_in_inference = False

    logger.info(f"Converting model to {onnx_model_save_path}")
    inp_random = torch.randn(1, 3, 416, 416)
    torch.onnx.export(
        model,
        inp_random,
        onnx_model_save_path,
        # export_params=True,
        # verbose=True,
        opset_version=11,
        # do_constant_folding=False,
        input_names=["images"],
        output_names=["pred_output"],
    )

    # check converted model
    logger.info("Checking converted model")
    onnx_model = onnx.load(onnx_model_save_path)
    onnx.checker.check_model(onnx_model)

    logger.info("All good")


if __name__ == "__main__":
    """Main entry point"""
    for model_code in MODEL_MAP.keys():
        convert_model(model_code)
