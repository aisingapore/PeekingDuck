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

"""
Detector class to handle detection of bboxes and masks for mask_rcnn
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch.backends as cudnn
from torch import Tensor
import torch.nn.functional as F
import torch

import numpy as np

from peekingduck.pipeline.nodes.model.yolact_edgev1.yolact_edge_files.model import (
    YolactEdge,
)
from peekingduck.pipeline.nodes.model.yolact_edgev1.yolact_edge_files.utils import (
    FastBaseTransform,
    crop,
)


class Detector:  # pylint: disable=too-many-instance-attributes
    """Detector class to handle detection of bboxes and masks for YolactEdge

    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): YolactEdge node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the Tensor
            will be allocated.
        yolact_edge (YolactEdge): The YolactEdge model for performing inference.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        class_names: List[str],
        detect_ids: List[int],
        model_type: str,
        num_classes: int,
        model_file: Dict[str, str],
        input_size: int,
        max_num_detections: int,
        score_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device_is_cuda: bool = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.device_is_cuda else "cpu")
        self.class_names = class_names
        self.detect_ids = detect_ids
        self.detect_ids_tensor = torch.tensor(
            detect_ids, dtype=torch.int64, device=self.device
        )
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_path = model_dir / model_file[self.model_type]
        self.input_size = (input_size, input_size)
        self.max_num_detections = max_num_detections
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.update_detect_ids(detect_ids)
        self.yolact_edge = self._create_yolact_edge_model()
        self.filtered_output: Dict[str, torch.Tensor] = dict()

    @torch.no_grad()
    def predict_instance_mask_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """YolactEdge masks and bboxes prediction function

        Args:
            image (np.ndarray): image in numpy array.

        Returns:
            bboxes (np.ndarray): array of detected bboxes
            labels (np.ndarray): array of labels
            scores (np.ndarray): array of scores
            masks (np.ndarray): array of detected masks
        """
        with torch.no_grad():
            if self.device_is_cuda:
                cudnn.benchmark = True
                cudnn.fastest = True
                cudnn.deterministic = True
                torch.set_default_tensor_type("torch.cuda.FloatTensor")
                frame = torch.from_numpy(image).cuda().float()
                torch.cuda.synchronize()
            else:
                torch.set_default_tensor_type("torch.FloatTensor")
                frame = torch.from_numpy(image).float()
        img_shape = image.shape[:2]
        model = self.yolact_edge

        preds = model(FastBaseTransform(self.input_size)(frame.unsqueeze(0)))[
            "pred_outs"
        ]
        labels, scores, boxes, masks = self._postprocess(preds[0], img_shape)

        return boxes, labels, scores, masks

    def update_detect_ids(self, ids: List[int]) -> None:
        """Updates list of selected object category IDs. When the list is
        empty, all available object category IDs are detected.

        Args:
            ids (List[int]): List of selected object category IDs
        """
        self.detect_ids = Tensor(ids).to(self.device)  # type: ignore

    def _create_yolact_edge_model(self) -> YolactEdge:
        """Creates YolactEdge model and loads its weights.
        Creates `detect_ids` as a `torch.Tensor`. Logs model configurations.

        Returns:
            (YolactEdge): YolactEdge model
        """
        self.logger.info(
            "YolactEdge model loaded with the following configs:\n\t"
            f"Model type: {self.model_type}\n\t"
            f"Input resolution: {self.input_size}\n\t"
            f"IDs being detected: {self.detect_ids.int().tolist()}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
            f"IOU threshold: {self.iou_threshold}"
        )

        return self._load_yolact_edge_weights()

    def _get_model(self) -> YolactEdge:
        """Constructs YolactEdge model based on parsed configuration.

        Returns:
            (YolactEdge): YolactEdge model.
        """
        return YolactEdge(
            self.model_type,
            self.input_size[0],
            self.iou_threshold,
            self.max_num_detections,
        )

    def _load_yolact_edge_weights(self) -> YolactEdge:
        """Loads YolactEdge model weights.

        Returns:
            (YolactEdge): YolactEdge model.

        Raises:
            ValueError: `model_path` does not exist.
        """
        if self.model_path.is_file():
            model = self._get_model()
            model.load_weights(self.model_path)
            model.eval()
            if self.device_is_cuda:
                model = model.cuda()
            return model

        raise ValueError(
            f"Model file does not exist. Please check that {self.model_path} exists"
        )

    def _postprocess(  # pylint: disable=too-many-locals
        self,
        network_output: Dict[str, Tensor],
        img_shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Postprocessing of detected bboxes and masks for YolactEdge

        Args:
            network_output (Dict[str, Tensor]): A dictionary from the first
                element of the YolactEdge output. The keys are "class", "box",
                "score", "mask", and "proto"
            img_shape (Tuple[int, int]): height and width of original image

        Returns:
            labels (ndarray): An array of human-friendly detection class names
            scores (ndarray): An array of confidence scores of the detections
            boxes (ndarray): An array of detection boxes of x1, y1, x2, y2 coords
            masks (ndarray): An array of masks in uint8
        """
        try:
            keep = network_output["score"] > self.score_threshold
            for k in network_output:
                if k != "proto":
                    network_output[k] = network_output[k][keep]

            detect_filter = torch.where(
                torch.isin(network_output["class"], self.detect_ids_tensor)
            )

            for output_key in network_output.keys():
                self.filtered_output[output_key] = network_output[output_key][
                    detect_filter
                ]

            classes = network_output["class"]
            box = network_output["box"]
            score = network_output["score"]
            mask = network_output["mask"]
            proto_data = network_output["proto"]

            mask = proto_data @ mask.t()
            mask = torch.sigmoid(mask)
            mask = crop(mask, box)
            mask = mask.permute(2, 0, 1).contiguous()
            mask = (
                F.interpolate(
                    mask.unsqueeze(0),
                    (img_shape[0], img_shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .gt_(0.5)
            )

            # Filters the detections to the IDs being detected as specified in the config
            labels = np.array([self.class_names[i] for i in classes[detect_filter]])
            boxes = np.array(box[detect_filter].cpu())
            boxes = np.clip(boxes, 0, 1)
            scores = np.array(score[detect_filter].cpu())
            masks = np.array(mask[detect_filter].cpu()).astype(np.uint8)

        except TypeError:
            return (
                np.empty((0)),
                np.empty((0), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 0, 0), dtype=np.uint8),
            )

        except RuntimeError:
            return (
                np.empty((0)),
                np.empty((0), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 0, 0), dtype=np.uint8),
            )

        return labels, scores, boxes, masks
