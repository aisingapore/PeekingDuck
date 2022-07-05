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

"""Transformation class for Generalized RCNN class.
Modifications include:
- Removed if conditions for keypoints detection
- Removed training / target related code and parameters
- Removed tracing related codes
- Removed torch_choice method (unused)
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union
import math
import torch
from torch import nn, Tensor
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection.roi_heads import (
    paste_masks_in_image,
)
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection.image_list import (
    ImageList,
)


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input resizing to match min_size / max_size

    It returns a ImageList for the inputs
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        min_size: int,
        max_size: int,
        image_mean: Iterable[float],
        image_std: Iterable[float],
        size_divisible: int = 32,
        fixed_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)  # type: ignore[assignment]
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size

    def forward(self, images: List[Tensor]) -> ImageList:
        """Normalizes and resizes the images, follow by padding the images to the same size and
        putting the images into the ImageList object that contains the padded images and the
        original image sizes

        Args:
            images (List[Tensor]): Input images

        Raises:
            ValueError: Raise error if the image does not have the shape [C, H, W]

        Returns:
            ImageList: An object containing padded images and the original image sizes
        """
        for i, image in enumerate(images):

            if image.dim() != 3:
                raise ValueError(
                    "images is expected to be a list of 3d tensors "
                    f"of shape [C, H, W], got {image.shape}"
                )
            image = self.normalize(image)
            image = self.resize(image)
            images[i] = image

        image_sizes = [img.shape[-2:] for img in images]
        padded_images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(padded_images, image_sizes_list)
        return image_list

    def normalize(self, image: Tensor) -> Tensor:
        """Normalize image"""
        if not image.is_floating_point():
            raise TypeError(
                "Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image: Tensor) -> Tensor:
        """Resize image"""
        # Note: min_size is either a list or tuple
        size = float(self.min_size[-1])  # type: ignore[index]
        image = _resize_image_and_masks(
            image, size, float(self.max_size), self.fixed_size
        )

        return image

    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        """Pad the images in a batch to the same size"""
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    @staticmethod
    def max_by_axis(the_list: List[List[int]]) -> List[int]:
        """Find the maximum size for each axis in a batch of images"""
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    @staticmethod
    def postprocess(
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        """Postprocess the result"""
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(result, image_shapes, original_image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks

        return result

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        indent = "\n    "
        format_string += (
            f"{indent}Normalize(mean={self.image_mean}, std={self.image_std})"
        )
        format_string += (
            f"{indent}Resize(min_size={self.min_size}, "
            f"max_size={self.max_size}, mode='bilinear')"
        )
        format_string += "\n)"
        return format_string


def resize_boxes(
    boxes: Tensor,
    original_size: Union[List[int], Tuple[int, int]],
    new_size: Union[List[int], Tuple[int, int]],
) -> Tensor:
    """Resize bounding boxes"""
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def _resize_image_and_masks(
    image: Tensor,
    self_min_size: float,
    self_max_size: float,
    fixed_size: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """Resize image and masks"""
    im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        scale_factor = scale.item()
        recompute_scale_factor = True

    image = nn.functional.interpolate(  # type: ignore[call-arg]
        image[None],
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False,
    )[0]

    return image
