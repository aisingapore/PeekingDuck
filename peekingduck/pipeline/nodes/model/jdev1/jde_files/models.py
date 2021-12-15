# Modifications copyright 2021 AI Singapore
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

"""
Create model backbone from config.
"""

from typing import Any, List, Dict, Optional, Tuple
from collections import OrderedDict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils.parse_config import (
    parse_model_cfg,
)
from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils.utils import (
    decode_delta_map,
)


# pylint: disable=too-many-locals
def create_modules(
    module_defs: List[Any],
) -> Tuple[Dict[Any, Any], nn.ModuleList]:
    """Constructs module list of layer blocks from module configuration
    in module_defs.

    Args:
        module_defs (List[Any]): Model configuration file.

    Returns:
        Tuple[Dict[Any, Any], nn.ModuleList]: Model layer blocks.
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            batch_normalize = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not batch_normalize,
                ),
            )
            if batch_normalize:
                batch_norm = nn.BatchNorm2d
                after_bn = batch_norm(filters)
                modules.add_module(f"batch_norm_{i}", after_bn)
                # BN is uniformly initialized by default in pytorch 1.0.1.
                # In pytorch>1.2.0, BN weights are initialized with constant 1,
                # but we find with the uniform initialization the model converges faster.
                nn.init.uniform_(after_bn.weight)
                nn.init.zeros_(after_bn.bias)
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module(f"maxpool_{i}", maxpool)
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]))
            modules.add_module(f"upsample_{i}", upsample)
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module(f"route_{i}", EmptyLayer())
        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module(f"shortcut_{i}", EmptyLayer())
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [float(x) for x in module_def["anchors"].split(",")]
            anchors = [
                (anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)  # type: ignore
            ]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = (int(hyperparams["width"]), int(hyperparams["height"]))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Define detection layer
            yolo_layer = YOLOLayer(
                anchors,  # type: ignore
                num_classes,
                int(hyperparams["nID"]),
                int(hyperparams["embedding_dim"]),
                img_size,
                yolo_layer_count,
                device,
            )
            modules.add_module(f"yolo_{i}", yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers."""

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()

    # pylint: disable=no-self-use
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return inputs


class Upsample(nn.Module):
    """Custom Upsample layer (nn.Upsample gives deprecated warning message)."""

    def __init__(self, scale_factor: int = 1, mode: str = "nearest") -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return F.interpolate(inputs, scale_factor=self.scale_factor, mode=self.mode)


# pylint: disable=too-many-instance-attributes, too-many-arguments, unused-argument, invalid-name, attribute-defined-outside-init
class YOLOLayer(nn.Module):
    """YOLO layer."""

    def __init__(
        self,
        anchors: List[Tuple[float, float]],
        num_classes: int,
        num_ids: int,
        num_emb_dim: int,
        img_size: Tuple[int, int],
        yolo_layer: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer = yolo_layer
        num_anchors = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_ids = num_ids
        self.img_size = 0
        self.emb_dim = num_emb_dim
        self.shift = [1, 3, 5]
        self.device = device

    def forward(
        self,
        p_cat: torch.Tensor,
        img_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            p_cat (torch.Tensor): Input from the previous layer.
            img_size (Tuple[int, int]): Size of image frame.

        Returns:
            (torch.Tensor): YOLO prediction tensor.
        """
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]

        if self.img_size != img_size:
            create_grids(self, img_size, nGh, nGw)

            if p.is_cuda:
                self.grid_xy = self.grid_xy.cuda()  # type: ignore
                self.anchor_wh = self.anchor_wh.cuda()  # type: ignore

        p = (
            p.view(nB, self.num_anchors, self.num_classes + 5, nGh, nGw)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # prediction

        p_emb = p_emb.permute(0, 2, 3, 1).contiguous()
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf

        p_conf = torch.softmax(p_conf, dim=1)[:, 1, ...].unsqueeze(-1)
        p_emb = F.normalize(
            p_emb.unsqueeze(1).repeat(1, self.num_anchors, 1, 1, 1).contiguous(), dim=-1
        )
        p_cls = torch.zeros(nB, self.num_anchors, nGh, nGw, 1).to(self.device)
        p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
        p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p), self.device)
        p[..., :4] *= self.stride

        return p.view(nB, -1, p.shape[-1])


class Darknet(nn.Module):
    """YOLOv3 object detection model."""

    def __init__(self, cfg_dict: Path, nID: int = 0, test_emb: bool = False) -> None:
        super().__init__()
        self.module_defs = parse_model_cfg(cfg_dict)
        self.module_defs[0]["nID"] = nID
        self.img_size = [
            int(self.module_defs[0]["width"]),
            int(self.module_defs[0]["height"]),
        ]
        self.emb_dim = int(self.module_defs[0]["embedding_dim"])
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.loss_names = ["loss", "box", "conf", "id", "nT"]
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.test_emb = test_emb

        self.classifier = nn.Linear(self.emb_dim, nID) if nID > 0 else None

    # pylint: disable=too-many-branches
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input from the previous layer.
            targets (torch.Tensor, optional): Targets tensor. Defaults to None.
            targets_len (int, optional): Length of targets. Defaults to None.

        Returns:
            torch.Tensor: Detection output tensor.
        """
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        layer_outputs: List[Any] = []
        output = []

        for _, (module_def, module) in enumerate(
            zip(self.module_defs, self.module_list)
        ):
            mtype = module_def["type"]
            if mtype in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif mtype == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == "shortcut":
                layer_i = int(module_def["from"])  # type: ignore
                x = layer_outputs[-1] + layer_outputs[layer_i]  # type: ignore
            elif mtype == "yolo":
                if self.test_emb:
                    if targets is not None:
                        targets = [  # type: ignore
                            targets[i][: int(l)] for i, l in enumerate(targets_len)  # type: ignore
                        ]
                    x = module[0](
                        x, self.img_size, targets, self.classifier, self.test_emb
                    )
                else:  # get detections
                    x = module[0](x, self.img_size)
                output.append(x)
            layer_outputs.append(x)

        # pylint: disable=no-else-return
        if self.test_emb:
            return torch.cat(output, 0)
        return torch.cat(output, 1)


def create_grids(self, img_size: Tuple[int, int], nGh: int, nGw: int) -> None:  # type: ignore
    """Create image grids.

    Args:
        img_size (Tuple[int, int]): Size of image frame.
        nGh (int): Number of grid height.
        nGw (int): Number of grid width.
    """
    self.stride = img_size[0] / nGw
    assert self.stride == img_size[1] / nGh, f"{self.stride} v.s. {img_size[1]}/{nGh}"

    # build xy offsets
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = (
        torch.arange(nGh)
        .repeat((nGw, 1))
        .transpose(0, 1)
        .view((1, 1, nGh, nGw))
        .float()
    )
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2)
