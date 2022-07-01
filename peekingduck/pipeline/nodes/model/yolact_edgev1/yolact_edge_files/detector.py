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

import torch.nn.functional as F
import numpy as np
import torch

from peekingduck.pipeline.nodes.model.yolact_edgev1.yolact_edge_files.model import YolactEdge
from peekingduck.pipeline.nodes.model.yolact_edgev1.yolact_edge_files.utils import (
    FastBaseTransform)

class Detector:
    """Detector class to handle detection of bboxes and masks for yolact_edge
    
    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): YolactEdge node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the torch.Tensor
            will be allocated.
        yolact_edge (YolactEdge): The YolactEdge model for performing inference.
    """
    def __init__(
        self,
        model_dir: Path,
        class_names: List[str],
        detect_ids: List[int],
        model_type: str,
        num_classes: int,
        model_file: Dict[str, str],
        input_size: int,
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = class_names
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_path = model_dir / model_file[self.model_type]
        self.input_size = (input_size, input_size)
        self.score_threshold = score_threshold

        self.update_detect_ids(detect_ids)
        self.yolact_edge = self._create_yolact_edge_model()

    @torch.no_grad()
    def predict_instance_mask_from_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict instance masks from image
        
        Args:
            image (np.ndarray): image in numpy array.
        
        Returns:
            bboxes (np.ndarray): array of detected bboxes
            labels (np.ndarray): array of labels
            scores (np.ndarray): array of scores
            masks (np.ndarray): array of detected masks
        """
        model = self.yolact_edge
        if torch.cuda.is_available():
            frame = torch.from_numpy(image).cuda().float()
        else:
            frame = torch.from_numpy(image).float()
        
        preds = model(FastBaseTransform()(frame.unsqueeze(0)))["pred_outs"]
        h, w, _ = image.shape

        labels, scores, bboxes, masks = postprocess(
            preds, w, h, score_threshold=self.score_threshold)
        return bboxes, labels, scores, masks

    def update_detect_ids(self, ids: List[int]) -> None:
        """Updates list of selected object category IDs. When the list is
        empty, all available object category IDs are detected.

        Args:
            ids: List of selected object category IDs
        """
        self.detect_ids = torch.Tensor(ids).to(self.device)  # type: ignore

    def _create_yolact_edge_model(self) -> YolactEdge:
        """Creates YolactEdge model
        
        Returns:
            YolactEdge: YolactEdge model
        """
        self.logger.info(
            "YolactEdge model loaded with the following configs:\n\t"
            f"Model type: {self.model_type}\n\t"
            f"Input resolution: {self.input_size}\n\t"
            f"IDs being detected: {self.detect_ids.int().tolist()}\n\t"
            f"Score threshold: {self.score_threshold}"
        )

        return self._load_yolact_edge_weights()

    def _get_model(self, model_type: str) -> YolactEdge:
        """Constructs YolactEdge model based on parsed configuration.
        Args:
            model_size (Dict[str, float]): Depth and width of the model.
        Returns:
            (YolactEdge): YolactEdge model.
        """
        return YolactEdge(
            self.model_type
        )

    def _load_yolact_edge_weights(self) -> YolactEdge:
        """Loads YolactEdge model weights.
        
        Args:
            model_path (Path): Path to model weights file.
            model_settings (Dict[str, float): Depth and width of the model.

        Returns:
            (YolactEdge): YolactEdge model.
        
        Raises:
            ValueError: `model_path` does not exist.
        """
        if self.model_path.is_file():
            model = self._get_model(self.model_type)
            model.load_weights(self.model_path)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            return model

        raise ValueError(
            f"Model file does not exist. Please check that {self.model_path} exists"
        )

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                crop_masks=True, score_threshold=0):
    dets = det_output[batch_idx]
    if dets is None:
        return [torch.Tensor()] * 4
    if score_threshold > 0:
        keep = dets['score'] > score_threshold
        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']
    proto_data = dets['proto']

    masks = proto_data @ masks.t()
    masks = torch.sigmoid(masks)

    masks = masks.permute(2, 0, 1).contiguous()
    masks = F.interpolate(masks.unsqueeze(0), (h, w), 
        mode=interpolation_mode, align_corners=False).squeeze(0)
    masks.gt_(0.5)

    classes = classes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()
    boxes = boxes.cpu().detach().numpy()

    masks = masks.cpu().detach().numpy()
    masks = masks.astype(np.uint8)

    return classes, scores, boxes, masks
