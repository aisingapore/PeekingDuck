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

"""Module to convert TensorFlow MoveNet models to TensorRT"""

from typing import Callable, Tuple
import logging
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert
from time import perf_counter


####################
# Globals
####################
# TensorRT precision modes lookup table
PRECISION_MODES = {
    "FP32": trt_convert.TrtPrecisionMode.FP32,
    "FP16": trt_convert.TrtPrecisionMode.FP16,
    "INT8": trt_convert.TrtPrecisionMode.INT8,
}
# Specify amount of GPU RAM to use for TRT computation (workspace in Nvidia jargon),
# remaining RAM is used for model.
# "Best" setting depends on GPU RAM available on device + model used
GPU_RAM_1G = 1000000000  # use as default in CONV_PARAMS below
GPU_RAM_2G = 2000000000
GPU_RAM_4G = 4000000000
GPU_RAM_6G = 6000000000
GPU_RAM_8G = 8000000000
# Model specification
MODEL_WEIGHTS_DIR = "../../peekingduck_weights"
MOVENET_MODEL_MAP = {
    "MPL": {
        "data_shape": (1, 256, 256, 3),
        "dir": MODEL_WEIGHTS_DIR + "/movenet/multipose_lightning",
    },
    "SPL": {
        "data_shape": (1, 192, 192, 3),
        "dir": MODEL_WEIGHTS_DIR + "/movenet/singlepose_lightning",
    },
    "SPT": {
        "data_shape": (1, 256, 256, 3),
        "dir": MODEL_WEIGHTS_DIR + "/movenet/singlepose_thunder",
    },
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_input_function(data_shape: Tuple[int, int, int, int]) -> Callable:
    """Generate the input function to be passed into TRT converter's build() method.

    Note: data_shape must be set to match model to be converted.
    """

    def my_input_fn() -> None:
        """Input function to be passed to converter's build() method"""
        inp = np.zeros(data_shape).astype(np.int32)
        yield (inp,)

    return my_input_fn


def convert(model_code: str, precision_mode: str, workspace_size: int) -> None:
    """Convert model weights in given model directory.
       Converted model output is saved directly to disk.

    Args:
        model_code (str): supported model codes { "MPL", "SPL", "SPT" }
        precision_mode (str): supported precision modes { "FP32", "FP16", "INT8" }
        workspace_size (int): TRT GPU workspace in bytes
    """
    assert model_code in MOVENET_MODEL_MAP.keys()
    assert precision_mode in PRECISION_MODES.keys()
    # prepare model working vars based on model's specifications
    data_shape = MOVENET_MODEL_MAP[model_code]["data_shape"]
    model_dir = MOVENET_MODEL_MAP[model_code]["dir"]
    build_input_fn = generate_input_function(data_shape)
    # prepare TRT working vars
    conv_params = trt_convert.TrtConversionParams(
        precision_mode=PRECISION_MODES[precision_mode],
        max_workspace_size_bytes=workspace_size,
    )
    converter = trt_convert.TrtGraphConverterV2(
        conversion_params=conv_params,
        input_saved_model_dir=model_dir,
    )
    model_out_dir = f"{model_dir}_{precision_mode.lower()}"

    logger.info(f"converting original model {model_code}...")
    start_time_convert = perf_counter()
    converter.convert()
    conv_dur = perf_counter() - start_time_convert
    logger.info(f"conversion time = {conv_dur:.2f} sec")

    logger.info("building generated model...")
    start_time_build = perf_counter()
    converter.build(input_fn=build_input_fn)
    build_dur = perf_counter() - start_time_build
    logger.info(f"build time = {build_dur:.2f} sec")

    logger.info(f"saving model to {model_out_dir}...")
    start_time_save = perf_counter()
    converter.save(model_out_dir)
    save_dur = perf_counter() - start_time_save
    logger.info(f"save time = {save_dur:.2f} sec")

    total_dur = perf_counter() - start_time_convert
    logger.info(f"Total time taken = {total_dur:.2f} sec")
    logger.info(f"Conversion time  = {conv_dur:.2f} sec")
    logger.info(f"Build time       = {build_dur:.2f} sec")
    logger.info(f"Save time        = {save_dur:.2f} sec")


if __name__ == "__main__":
    """Main entry point for converting TF to TRT"""
    # Use FP16 'coz INT8 requires special training and is generally not directly usable.
    # Default to use 1GB of GPU RAM for TRT workspace.
    # (testing with >= 2GB doesn't increase model FPS on available hardware)
    convert("MPL", "FP16", GPU_RAM_1G)
    convert("SPL", "FP16", GPU_RAM_1G)
    convert("SPT", "FP16", GPU_RAM_1G)
