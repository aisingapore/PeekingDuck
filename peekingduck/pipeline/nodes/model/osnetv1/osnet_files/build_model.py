# Copyright 2021 AI Singapore
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
# Original copyright (c) 2018 Kaiyang Zhou
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
Build OSNet model.
"""

from __future__ import absolute_import
import torch
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.osnet import osnet_x1_0
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.osnet_ain import (
    osnet_ain_x1_0,
)


__model_factory = {
    "osnet_x1_0": osnet_x1_0,
    "osnet_ain_x1_0": osnet_ain_x1_0,
}


def build_model(
    name: str,
    num_classes: int,
    loss: str = "softmax",
    use_gpu: bool = True,
) -> torch.nn.Module:
    """A function wrapper for building a model.

    Args:
        name (str): Model name.
        num_classes (int): Number of training identities.
        loss (str): Loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        use_gpu (bool): Whether to use gpu. Default is True.

    Returns:
        nn.Module
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(f"Unknown model: {name}. Must be one of {avai_models}")
    return __model_factory[name](num_classes=num_classes, loss=loss, use_gpu=use_gpu)
