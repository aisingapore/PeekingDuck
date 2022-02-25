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
# Original copyright (c) 2020 YifuZhang
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

"""Decodes model output to bbox and indices.

Modifications:
- Refactor mot_decode() to a class instead
- Remove unnecessary creation of lists since batch size 1 is hardcoded
- Hardcode ctdet_post_process to handle detections of batch size 1 since this
    assumptions is already made when calling the function
- Hardcode num_classes=1
    - Change post_process output type
    - Change ctdet_post_process output type
- Remove unnecessary .tolist() in ctdet_post_process
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from peekingduck.pipeline.nodes.model.fairmotv1.fairmot_files.utils import (
    gather_feat,
    transform_coords,
    transpose_and_gather_feat,
)


class Decoder:  # pylint: disable=too-few-public-methods
    """Decodes model output to bounding box coordinates and indices following
    the approach adopted by CenterNet.
    """

    def __init__(self, max_per_image: int, down_ratio: int) -> None:
        self.max_per_image = max_per_image
        self.down_ratio = down_ratio

    def __call__(  # pylint: disable=too-many-arguments
        self,
        heatmap: torch.Tensor,
        size: torch.Tensor,
        offset: torch.Tensor,
        orig_shape: Tuple[int, ...],
        input_shape: torch.Size,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Decodes model outputs to bounding box coordinates and indices.

        Args:
            heatmap (torch.Tensor): A heatmap predicting where the object
                center will be.
            size (torch.Tensor): Size of the bounding boxes w.r.t. the object
                centers.
            offset (torch.Tensor): A continuous offset relative to the object
                centers to localize objects more precisely.
            orig_shape (Tuple[int, ...]): Shape of the original image.
            input_shape (torch.Size): Shape of the image fed to the model.

        Returns:
            (Tuple[np.ndarray, torch.Tensor]): A tuple containing detections
            and their respective indices. Indices are used to filter the Re-ID
            feature tensor.
        """
        batch, _, _, _ = heatmap.size()

        heatmap = self._nms(heatmap)

        scores, indices, classes, y_coords, x_coords = self._topk(heatmap)
        offset = transpose_and_gather_feat(offset, indices)
        offset = offset.view(batch, self.max_per_image, 2)
        x_coords = x_coords.view(batch, self.max_per_image, 1) + offset[:, :, 0:1]
        y_coords = y_coords.view(batch, self.max_per_image, 1) + offset[:, :, 1:2]
        size = transpose_and_gather_feat(size, indices)
        size = size.view(batch, self.max_per_image, 4)
        classes = classes.view(batch, self.max_per_image, 1)
        scores = scores.view(batch, self.max_per_image, 1)
        bboxes = torch.cat(
            [
                x_coords - size[..., 0:1],
                y_coords - size[..., 1:2],
                x_coords + size[..., 2:3],
                y_coords + size[..., 3:4],
            ],
            dim=2,
        )
        detections = torch.cat([bboxes, scores, classes], dim=2)
        detections = self._post_process(detections, orig_shape, input_shape)
        # Currently not needed as FairMOT only detect one class
        # detections = self._trim_outputs(detections)

        return detections, indices

    def _post_process(
        self,
        detections: torch.Tensor,
        orig_shape: Tuple[int, ...],
        input_shape: torch.Size,
    ) -> np.ndarray:
        """Post processes the detections following the approach by CenterNet.
        Translates/scales detections w.r.t. original image shape.

        Args:
            detections (torch.Tensor): Detections with the format
                [x1, y1, x2, y2, score, class] where (x1, y1) is top left and
                (x2, y2) is bottom right.
            orig_shape (Tuple[int, ...]): Shape of the original image.
            input_shape (torch.Size): Shape of the image fed to the model.

        Returns:
            (np.ndarray): Transformed detections w.r.t. the original image
            shape.
        """
        orig_h = float(orig_shape[0])
        orig_w = float(orig_shape[1])
        input_h = float(input_shape[2])
        input_w = float(input_shape[3])

        dets = detections.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        # Detection batch size has been hardcoded to 1 in FairMOT
        return _ctdet_post_process(
            dets[0].copy(),
            np.array([orig_w / 2.0, orig_h / 2.0], dtype=np.float32),
            max(input_w / input_h * orig_h, orig_w),
            (input_w // self.down_ratio, input_h // self.down_ratio),
        )

    def _topk(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects top k scores and decodes to get xy coordinates.

        Args:
            scores (torch.Tensor): In the case of FairMOT, this is a heatmap
                predicting where the object center will be.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor]): Tuple containing top k detection scores and their
            respective indices, classes, y-, and x- coordinates.
        """
        k = self.max_per_image
        batch, cat, height, width = scores.size()

        topk_scores, topk_indices = torch.topk(scores.view(batch, cat, -1), k)

        topk_indices = (topk_indices % (height * width)).view(batch, -1, 1)
        topk_y_coords = torch.div(topk_indices, width, rounding_mode="floor")
        topk_x_coords = topk_indices % width

        topk_score, topk_index = torch.topk(topk_scores.view(batch, -1), k)
        topk_classes = torch.div(topk_index, k, rounding_mode="trunc")

        topk_indices = gather_feat(topk_indices, topk_index).view(batch, k)
        topk_y_coords = gather_feat(topk_y_coords, topk_index).view(batch, k)
        topk_x_coords = gather_feat(topk_x_coords, topk_index).view(batch, k)

        return topk_score, topk_indices, topk_classes, topk_y_coords, topk_x_coords

    def _trim_outputs(self, detections: np.ndarray) -> np.ndarray:  # pragma: no cover
        """In the case of multi-class detections, trims the output to be
        <=`self.max_per_image`.

        Args:
            detections (np.ndarray): Object detection results.

        Returns:
            (np.ndarray): Trimmed detection results.
        """
        if len(detections) > self.max_per_image:
            # FairMOT only detects one class so this is never called.
            kth = len(detections) - self.max_per_image
            cut_off = np.partition(detections[:, 4], kth)[kth]
            detections = detections[detections[:, 4] >= cut_off]
        return detections

    @staticmethod
    def _nms(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
        """Uses maxpool to filter the max score and get local peaks.

        Args:
            heatmap (torch.Tensor): A heatmap predicting where the object
                center will be.
            kernel (int): Size of the window to take a max over.

        Returns:
            (torch.Tensor): Heatmap with only local peaks remaining.
        """
        pad = (kernel - 1) // 2

        hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep


def _ctdet_post_process(
    detections: np.ndarray,
    center: np.ndarray,
    scale: float,
    output_size: Tuple[float, float],
) -> np.ndarray:
    """Post-processes detections and translate/scale it back to the original
    image.

    Args:
        detections (np.ndarray): An array of detections each having the format
            [x1, y1, x2, y2, score, class] where (x1, y1) is top left and
            (x2, y2) is bottom right.
        center (np.ndarray): Coordinate of the center of the original image.
        scale (float): Scale between original image and input image fed to the
            model.
        output_size (Tuple[float, float]): Size of output by the model.

    Returns:
        (np.ndarray): Detections with coordinates transformed w.r.t. the
        original image.
    """
    detections[:, :2] = transform_coords(detections[:, :2], center, scale, output_size)
    detections[:, 2:4] = transform_coords(
        detections[:, 2:4], center, scale, output_size
    )
    classes = detections[:, -1]
    mask = classes == 0

    return np.concatenate([detections[mask, :4], detections[mask, 4:5]], axis=1).astype(
        np.float32
    )
