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
# Original Copyright From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America
#           (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute
#           (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Generate a backbone with FPN for Mask-RCNN model.

Modifications include:
- Removed unused arguments:
    - resnet_fpn_backbone
        - trainable_layers
        - returned_layers
        - extra_blocks
        - norm_layer
"""

from typing import Dict, Optional, List
from collections import OrderedDict
from torch import nn, Tensor
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.ops import (
    misc,
    feature_pyramid_network as fpn,
)
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.backbones import (
    resnet,
)


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses IntermediateLayerGetter to extract a submodel that returns
    the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By
            default a ``LastLevelMaxPool`` is used.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[fpn.ExtraFPNBlock] = None,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = fpn.LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = fpn.FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Takes the input tensor and pass it through the intermediate layers of the model.
        Then, the output from the intermediate layers will be passed to the FPN before returning
        the final output.

        Args:
            inputs (Tensor): Input feature

        Returns:
            Dict[str, Tensor]: An output dictionary. The keys are the names of the returned
                activations of the backbone, and the values are the output features from the FPN
                layers ordered from highest resolution first.
        """
        x_out = self.body(inputs)
        x_out = self.fpn(x_out)
        return x_out


def resnet_fpn_backbone(backbone_name: str) -> BackboneWithFPN:
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of
    layers in the backbone.

    Args:
        backbone_name (string): resnet architecture. Possible values are 'resnet50', 'resnet101'

    Returns:
        BackboneWithFPN: An object of the backbone with the FPN attached
    """
    backbone = resnet.__dict__[backbone_name](norm_layer=misc.FrozenBatchNorm2d)

    # select layers that wont be frozen
    for _, parameter in backbone.named_parameters():
        parameter.requires_grad_(False)

    extra_blocks = fpn.LastLevelMaxPool()

    returned_layers = [1, 2, 3, 4]
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    # backbone.inplanes will be equal to the final output feature channels (2048 for both r50
    # and r101), and dividing by 8 will be equal to the number of output channels at the conv2_x
    # layer (layer 1) (256 for both r50 and r101). The elements in the in_channels_list (list of
    # number of input channels for the FPN) should be equal to the number of output channels for
    # the respective layers in the returned_layers, which is increased by a factor of 2 for every
    # subsequent layers (e.g. layer 1 to 4 --> [256, 512, 1024, 2048] for R-50 and R-101).
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=extra_blocks,
    )


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Takes in a Tensor and obtain the outputs from the respective layers specified in
        self.return_layers.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Dict[str, Tensor]: A dictionary with keys equal to the enumerated numbers of the
                `return_layers`, and the value is the output features from the respective
                intermediate layers specified in self.return_layers
        """
        out = OrderedDict()
        for name, module in self.items():
            inputs = module(inputs)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = inputs
        return out
