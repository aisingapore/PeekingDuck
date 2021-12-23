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
API to extract features.
"""

from __future__ import absolute_import
from typing import List, Tuple, Union
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.build_model import build_model
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.torchtools import (
    check_isfile,
    load_pretrained_weights,
)


class FeatureExtractor:  # pylint: disable=too-few-public-methods
    """A simple API for feature extraction.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).

    Returns:
        torch.Tensor: with shape (B, D) where D is the feature dimension.
    """

    # pylint: disable=dangerous-default-value, too-many-arguments
    def __init__(
        self,
        model_name: str = "",
        model_path: str = "",
        image_size: Tuple[int, int] = (256, 128),
        pixel_mean: List = [0.485, 0.456, 0.406],
        pixel_std: List = [0.229, 0.224, 0.225],
        pixel_norm: bool = True,
        device: str = "cuda",
    ) -> None:
        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            use_gpu=device.startswith("cuda"),
        )
        model.eval()

        model_fpath = Path(model_path)
        if model_fpath and check_isfile(model_fpath):
            load_pretrained_weights(model, model_fpath)

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)  # type: ignore
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(  # pylint: disable=redefined-builtin
        self, input: Union[List[Union[str, np.ndarray]], str, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert("RGB")

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        "Type of each element must belong to [str | numpy.ndarray]"
                    )

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)  # type: ignore
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert("RGB")
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)  # type: ignore

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features
