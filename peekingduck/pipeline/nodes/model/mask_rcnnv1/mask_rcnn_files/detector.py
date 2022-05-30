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
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import detection
from torchvision.models.detection.mask_rcnn import MaskRCNN
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn


class Detector:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Detector class to handle detection of bboxes and masks for mask_rcnn"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        class_names: Dict[int, str],
        detect_ids: List[int],
        model_type: str,
        num_classes: int,
        model_file: Dict[str, str],
        min_size: int,
        max_size: int,
        nms_iou_threshold: float,
        max_num_detections: int,
        score_threshold: float,
        mask_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.class_names = class_names
        self.detect_ids = detect_ids
        self.detect_ids_set = set(detect_ids)
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_path = model_dir / model_file[self.model_type]
        self.min_size = min_size
        self.max_size = max_size
        self.nms_iou_threshold = nms_iou_threshold
        self.max_num_detections = max_num_detections
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_map = {"r50-fpn": "resnet50"}
        self.preprocess_transform = T.Compose(
            [
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
            ]
        )
        self.mask_rcnn = self._create_mask_rcnn_model()

    @torch.no_grad()
    def predict_instance_mask_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mask R-CNN masks and bboxes prediction function

        Args:
            image (np.ndarray): image in numpy array

        Returns:
            bboxes (np.ndarray): array of detected bboxes
            labels (np.ndarray): array of labels
            scores (np.ndarray): array of scores
            masks (np.ndarray): array of detected masks
        """
        img_shape = image.shape[:2]
        processed_image = self._preprocess(image)
        # run network
        network_output = self.mask_rcnn(processed_image)
        bboxes, labels, scores, masks = self._postprocess(network_output, img_shape)

        return bboxes, labels, scores, masks

    def _create_mask_rcnn_model(self) -> torch.nn.Module:
        """Initialize the Mask R-CNN model with the config settings and pretrained weights

        Returns:
            torch.nn.Module: An initialized model in PyTorch framework
        """
        trainable_backbone_layers = 0
        backbone_name = self.model_name_map[self.model_type]
        backbone = detection.backbone_utils.resnet_fpn_backbone(
            backbone_name=backbone_name,
            pretrained=True,
            trainable_layers=trainable_backbone_layers,
        )
        model = MaskRCNN(
            backbone=backbone,
            num_classes=self.num_classes,
            box_nms_thresh=self.nms_iou_threshold,
            box_score_thresh=self.score_threshold,
            box_detections_per_img=self.max_num_detections,
            min_size=self.min_size,
            max_size=self.max_size,
        )
        if self.model_path.is_file():
            state_dict = torch.load(self.model_path, map_location=self.device)
        else:
            raise FileNotFoundError(
                f"Model file does not exist. Please check that {self.model_path} exists."
            )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        self.logger.info(
            "Mask-RCNN model loaded with following configs:\n\t"
            f"Model type: {self.model_type}\n\t"
            f"IDs being detected: {self.detect_ids}\n\t"
            f"NMS IOU threshold: {self.nms_iou_threshold}\n\t"
            f"Maximum number of detections per image: {self.max_num_detections}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
            f"Maximum size of the image: {self.max_size}\n\t"
            f"Minimum size of the image: {self.min_size}"
        )
        return model

    def _postprocess(
        self,
        network_output: List[Dict[str, Any]],
        img_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Postprocessing of detected bboxes and masks for mask_rcnn

        Args:
            network_output (list): list of boxes, scores and labels from network
            img_shape (Tuple[int, int]): height of original image

        Returns:
            boxes (np.ndarray): postprocessed array of detected bboxes
            scores (np.ndarray): postprocessed array of scores
            labels (np.ndarray): postprocessed array of labels
            masks (np.ndarray): postprocessed array of binarized masks
        """
        masks = np.empty((0, 0, 0), dtype=np.uint8)
        scores = np.empty((0), dtype=np.float32)
        labels = np.empty((0))
        bboxes = np.empty((0, 4), dtype=np.float32)
        if len(network_output[0]["labels"]) > 0:
            network_output[0]["labels"] -= 1

            detect_filter = []
            # Indices to filter out unwanted classes
            for label in network_output[0]["labels"]:
                detect_filter.append(label.item() in self.detect_ids_set)

            for output_key in network_output[0].keys():
                network_output[0][output_key] = network_output[0][output_key][
                    detect_filter
                ]

            if len(network_output[0]["labels"]) > 0:
                bboxes = network_output[0]["boxes"].cpu().numpy()
                # Normalize the bbox coordinates by the image size
                bboxes = xyxy2xyxyn(bboxes, height=img_shape[0], width=img_shape[1])
                bboxes = np.clip(bboxes, 0, 1)

                label_numbers = network_output[0]["labels"].cpu().numpy()
                labels = np.vectorize(self.class_names.get)(label_numbers)

                scores = network_output[0]["scores"].cpu().numpy()

                # Binarize mask's pixel values by confidence score
                masks = network_output[0]["masks"] > self.mask_threshold
                masks = masks.squeeze(1).cpu().numpy().astype(np.uint8)

        return bboxes, labels, scores, masks

    def _preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """Preprocessing function for mask_rcnn

        Args:
            image (np.ndarray): Image in numpy array.

        Returns:
            image (np.ndarray): The preprocessed image
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return [self.preprocess_transform(image_rgb).to(self.device)]
