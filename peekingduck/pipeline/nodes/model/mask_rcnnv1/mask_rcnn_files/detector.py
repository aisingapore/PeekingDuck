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
import cv2
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as T
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection.backbone_utils import (
    resnet_fpn_backbone,
)
from peekingduck.pipeline.nodes.model.mask_rcnnv1.mask_rcnn_files.detection.mask_rcnn import (
    MaskRCNN,
)


class Detector:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Detector class to handle detection of bboxes and masks for mask_rcnn"""

    preprocess_transform = T.Compose(
        [
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
        ]
    )
    model_name_map = {"r50-fpn": "resnet50", "r101-fpn": "resnet101"}

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
        iou_threshold: float,
        max_num_detections: int,
        score_threshold: float,
        mask_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.detect_ids = torch.tensor(
            detect_ids, dtype=torch.int64, device=self.device
        )
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_path = model_dir / model_file[self.model_type]
        self.min_size = min_size
        self.max_size = max_size
        self.iou_threshold = iou_threshold
        self.max_num_detections = max_num_detections
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.mask_rcnn = self._create_mask_rcnn_model()
        self.filtered_output: Dict[str, Tensor] = {}

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
        processed_images = self._preprocess(image)
        # run network, assumes batch size of 1 for peekingduck inference
        network_output = self.mask_rcnn(processed_images)[0]
        bboxes, labels, scores, masks = self._postprocess(network_output, img_shape)

        return bboxes, labels, scores, masks

    def _create_mask_rcnn_model(self) -> MaskRCNN:
        """Creates a Mask-RCNN model and loads its weights. It also logs model configurations.

        Returns:
            (MaskRCNN): An initialized and loaded model in PyTorch framework
        """
        self.logger.info(
            "Mask-RCNN model loaded with following configs:\n\t"
            f"Model type: {self.model_type}\n\t"
            f"IDs being detected: {self.detect_ids.tolist()}\n\t"
            f"IOU threshold: {self.iou_threshold}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
            f"Mask threshold: {self.mask_threshold}\n\t"
            f"Maximum number of detections per image: {self.max_num_detections}\n\t"
            f"Maximum size of the image: {self.max_size}\n\t"
            f"Minimum size of the image: {self.min_size}"
        )

        return self._load_mask_rcnn_weights()

    def _get_model(self) -> MaskRCNN:
        """Constructs Mask-RCNN model based on parsed configuration.

        Returns:
            (MaskRCNN): Mask-RCNN model.
        """
        backbone_name = Detector.model_name_map[self.model_type]
        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
        )
        return MaskRCNN(
            backbone=backbone,
            num_classes=self.num_classes,
            box_nms_thresh=self.iou_threshold,
            box_score_thresh=self.score_threshold,
            box_detections_per_img=self.max_num_detections,
            min_size=self.min_size,
            max_size=self.max_size,
        )

    def _load_mask_rcnn_weights(self) -> MaskRCNN:
        """Loads Mask-RCNN model weights

        Raises:
            FileNotFoundError: Raise error when model path does not exist

        Returns:
            (MaskRCNN): Mask-RCNN model loaded with weights
        """
        if self.model_path.is_file():
            state_dict = torch.load(self.model_path, map_location=self.device)
            model = self._get_model()
            model.load_state_dict(state_dict)
            model.eval().to(self.device)
            return model

        raise FileNotFoundError(
            f"Model file does not exist. Please check that {self.model_path} exists."
        )

    def _postprocess(
        self,
        network_output: Dict[str, Tensor],
        img_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Postprocessing of detected bboxes and masks for mask_rcnn

        Args:
            network_output (Dict[str, Tensor]): A dictionary from the first element of the
                Mask-RCNN output. The keys are should have "labels", "boxes", "scores" and "masks"
            img_shape (Tuple[int, int]): height and width of original image

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
        if len(network_output["labels"]) > 0:
            network_output["labels"] -= 1

            # Indices to filter out unwanted classes
            detect_filter = torch.where(
                torch.isin(network_output["labels"], self.detect_ids)
            )

            for output_key in network_output.keys():
                self.filtered_output[output_key] = network_output[output_key][
                    detect_filter
                ]

            if len(self.filtered_output["labels"]) > 0:
                bboxes = self.filtered_output["boxes"].cpu().numpy()
                # Normalize the bbox coordinates by the image size
                bboxes = xyxy2xyxyn(bboxes, height=img_shape[0], width=img_shape[1])
                bboxes = np.clip(bboxes, 0, 1)

                label_numbers = self.filtered_output["labels"].cpu().numpy()
                labels = np.vectorize(self.class_names.get)(label_numbers)

                scores = self.filtered_output["scores"].cpu().numpy()

                # Binarize mask's pixel values by confidence score
                masks = self.filtered_output["masks"] > self.mask_threshold
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
        return [Detector.preprocess_transform(image_rgb).to(self.device)]
