# Modifications copyright 2022 AI Singapore
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


def strip_optimizer(
    weights_path: Path, key_to_keep: str = "model", half: bool = False
) -> None:  # pragma: no cover
    """Removes non-model items from the weights file.

    The inference Detector requires only "model" parameters from the weights
    file. Deletes all other values from the weights file. Optionally, converts
    all float and double parameters to half-precision. Leaves int parameters
    such as `num_batches_tracked` in `BatchNorm2d` untouched.

    Args:
        weights_path (Path): Path to weights file.
        key_to_keep (str): The key which contains the models weights. Default
            is "model".
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
    delete_keys = [key for key in ckpt.keys() if key != key_to_keep]
    for key in delete_keys:
        del ckpt[key]
    for param in ckpt[key_to_keep]:
        # Only convert double and float to half-precision
        if half and ckpt[key_to_keep][param].dtype in (torch.double, torch.float):
            ckpt[key_to_keep][param] = ckpt[key_to_keep][param].half()
        ckpt[key_to_keep][param].requires_grad = False

    torch.save(ckpt, str(stripped_weights_path))
    stripped_filesize = stripped_weights_path.stat().st_size / 1e6
    print(
        f"Saved as {stripped_weights_path}. "
        f"Original size: {orig_filesize}MB. "
        f"Stripped size: {stripped_filesize}MB."
    )
