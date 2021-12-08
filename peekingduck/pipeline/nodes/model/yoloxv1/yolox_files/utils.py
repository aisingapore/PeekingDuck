# Modifications copyright 2021 AI Singapore
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
#
# Original copyright 2021 Megvii, Base Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of utility functions used by YOLOX.

Modifications include:
- Refactor fuse_conv_and_bn and fuse_model
- Adding xywh2xyxy and xyxy2xyxyn
"""

from pathlib import Path
from typing import no_type_check

import numpy as np
import torch
import torch.nn as nn

from peekingduck.pipeline.nodes.model.yoloxv1.yolox_files.model import YOLOX
from peekingduck.pipeline.nodes.model.yoloxv1.yolox_files.network_blocks import BaseConv


@no_type_check
def fuse_conv_and_bn(conv: nn.Conv2d, batch_norm: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuses convolution and batchnorm layers.

    Reference: https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """
    fusedconv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        groups=conv.groups,
        bias=True,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    ).requires_grad_(False)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(
        batch_norm.weight.div(torch.sqrt(batch_norm.eps + batch_norm.running_var))
    )
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(
            conv.weight.size(0), dtype=conv.weight.dtype, device=conv.weight.device
        )
        if conv.bias is None
        else conv.bias
    )
    b_bn = batch_norm.bias - batch_norm.weight.mul(batch_norm.running_mean).div(
        torch.sqrt(batch_norm.running_var + batch_norm.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model: YOLOX) -> YOLOX:
    """Fuses the batch normalization layers in `BaseConv` modules."""
    for module in model.modules():
        if isinstance(module, BaseConv) and hasattr(module, "bn"):
            module.conv = fuse_conv_and_bn(module.conv, module.bn)  # update conv
            delattr(module, "bn")  # remove batchnorm
            # update forward
            module.forward = module.fuseforward  # type: ignore
    return model


def strip_optimizer(weights_path: Path, half: bool = False) -> None:  # pragma: no cover
    """Removes non-model items from the weights file.

    The inference Detector requires only "model" parameters from the weights
    file. Deletes all other values from the weights file. Optionally, converts
    all float and double parameters to half-precision. Leaves int parameters
    such as `num_batches_tracked` in `BatchNorm2d` untouched.

    Args:
        weights_path (Path): Path to weights file.
        half (bool): Flag to determine if float and double parameters should
            be converted to half-precision.
    """
    stripped_weights_path = weights_path.with_name(
        f"{weights_path.stem}-stripped"
        f"{'-half' if half else ''}{weights_path.suffix}"
    )
    orig_filesize = weights_path.stat().st_size / 1e6
    ckpt = torch.load(str(weights_path), map_location=torch.device("cpu"))
    # Remove all data other than "model", such as amp, optimizer, start_epoch
    delete_keys = [key for key in ckpt.keys() if key != "model"]
    for key in delete_keys:
        del ckpt[key]
    for param in ckpt["model"]:
        # Only convert double and float to half-precision
        if half and ckpt["model"][param].dtype in (torch.double, torch.float):
            ckpt["model"][param] = ckpt["model"][param].half()
        ckpt["model"][param].requires_grad = False

    torch.save(ckpt, str(stripped_weights_path))
    stripped_filesize = stripped_weights_path.stat().st_size / 1e6
    print(
        f"Saved as {stripped_weights_path}. "
        f"Original size: {orig_filesize}MB. "
        f"Stripped size: {stripped_filesize}MB."
    )


def xywh2xyxy(inputs: torch.Tensor) -> torch.Tensor:
    """Converts from [x, y, w, h] to [x1, y1, x2, y2] format.

    (x, y) is the object center. (x1, y1) is the top left corner and (x2, y2)
    is the bottom right corner.
    """
    outputs = torch.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] - inputs[:, 2] / 2
    outputs[:, 1] = inputs[:, 1] - inputs[:, 3] / 2
    outputs[:, 2] = inputs[:, 0] + inputs[:, 2] / 2
    outputs[:, 3] = inputs[:, 1] + inputs[:, 3] / 2

    return outputs


def xyxy2xyxyn(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    """Converts from [x1, y1, x2, y2] to normalised [x1, y1, x2, y2].

    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.
    Normalised coordinates are w.r.t. original image size.
    """
    outputs = np.empty_like(inputs)
    outputs[:, [0, 2]] = inputs[:, [0, 2]] / width
    outputs[:, [1, 3]] = inputs[:, [1, 3]] / height

    return outputs
