# Modifications copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original copyright (c) 2019 ZhongdaoWang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Darknet-53 backbone for JDE model.

Modifications include:
- Remove training related code such as:
    - classifier
    - loss_names
    - losses
    - test_emb
    - uniform initialization of batch norm
- Refactor to remove unused code
    - enumerate in forward()
    - "maxpool" in _create_nodes
- Refactor for proper type hinting
    - renamed one of layer_i to layer_indices in forward()
- Refactor in _create_nodes to reduce the number of local variables
- Use the nn.Upsample instead of the custom one since it no longer gives
    deprecated warning
- Use nn.Identity instead of custom EmptyLayer
    - relevant PR: https://github.com/pytorch/pytorch/pull/19249
- Removed yolo_layer_count since layer member variable has been removed in
    YOLOLayer as it's not used
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from peekingduck.pipeline.nodes.model.jdev1.jde_files.network_blocks import YOLOLayer


class Darknet(nn.Module):
    """YOLOv3 object detection model.

    Args:
        cfg_dict (List[Dict[str, Any]]): Model architecture
            configurations.
        device (torch.device): The device which a `torch.Tensor` is on or
            will be allocated.
        num_identities (int): Number of identities, e.g., number of unique
            pedestrians. Uses 14455 for JDE according to the original code.
    """

    def __init__(
        self, cfg_dict: List[Dict[str, Any]], device: torch.device, num_identities: int
    ) -> None:
        super().__init__()
        self.module_defs = cfg_dict
        self.device = device
        self.module_defs[0]["nID"] = num_identities
        self.img_size = (
            int(self.module_defs[0]["width"]),
            int(self.module_defs[0]["height"]),
        )
        self.emb_dim = int(self.module_defs[0]["embedding_dim"])
        self.hyperparams, self.module_list = _create_modules(
            self.module_defs, self.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Input from the previous layer.

        Returns:
            (torch.Tensor): A dictionary of tensors with keys corresponding to
                `self.out_features`.
        """
        layer_outputs: List[torch.Tensor] = []
        outputs = []

        for module_def, module in zip(self.module_defs, self.module_list):
            module_type = module_def["type"]
            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_type == "route":
                layer_indices = list(map(int, module_def["layers"].split(",")))
                if len(layer_indices) == 1:
                    x = layer_outputs[layer_indices[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_indices], 1)
            elif module_type == "shortcut":
                x = layer_outputs[-1] + layer_outputs[int(module_def["from"])]
            elif module_type == "yolo":
                x = module[0](x, self.img_size)
                outputs.append(x)
            layer_outputs.append(x)

        return torch.cat(outputs, 1)


def _create_modules(
    module_defs: List[Dict[str, Any]], device: torch.device
) -> Tuple[Dict[str, Any], nn.ModuleList]:
    """Constructs module list of layer blocks from module configuration in
    `module_defs`

    NOTE: Each `module_def` in `module_defs` is parsed as a dictionary
    containing string values. As a result, "1" can sometimes represent True
    instead of the number of the key. We try to do == "1" instead of implicit
    boolean by converting it to int.

    Args:
        module_defs (List[Dict[str, Any]]): A list of module definitions.
        device (torch.device): The device which a `torch.Tensor` is on or
                will be allocated.

    Returns:
        (Tuple[Dict[str, Any], nn.ModuleList]): A tuple containing a dictionary
            of model hyperparameters and a ModuleList containing the modules
            in the model.
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        module_type = module_def["type"]
        modules = nn.Sequential()
        if module_type == "convolutional":
            has_batch_norm = module_def["batch_normalize"] == "1"
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            modules.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=(kernel_size - 1) // 2 if module_def["pad"] == "1" else 0,
                    bias=not has_batch_norm,
                ),
            )
            if has_batch_norm:
                modules.add_module(f"batch_norm_{i}", nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))
        elif module_type == "upsample":
            modules.add_module(
                f"upsample_{i}",
                nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest"),
            )
        elif module_type == "route":
            filters = sum(
                [
                    output_filters[i + 1 if i > 0 else i]
                    for i in map(int, module_def["layers"].split(","))
                ]
            )
            modules.add_module(f"route_{i}", nn.Identity())
        elif module_type == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module(f"shortcut_{i}", nn.Identity())
        elif module_type == "yolo":
            # Extract anchors
            anchor_dims = iter(map(float, module_def["anchors"].split(",")))
            # This lets us do pairwise() with no overlaps
            anchors = list(zip(anchor_dims, anchor_dims))
            anchors = [anchors[i] for i in map(int, module_def["mask"].split(","))]
            # Define detection layer
            modules.add_module(
                f"yolo_{i}",
                YOLOLayer(
                    anchors,
                    int(module_def["classes"]),
                    int(hyperparams["nID"]),
                    int(hyperparams["embedding_dim"]),
                    device,
                ),
            )

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list
