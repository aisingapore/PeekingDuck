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

"""Detector module to predict object bbox from an image using YOLOX."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision

from peekingduck.pipeline.nodes.model.yoloxv1.yolox_files.model import YOLOX
from peekingduck.pipeline.nodes.model.yoloxv1.yolox_files.utils import (
    fuse_model,
    xywh2xyxy,
    xyxy2xyxyn,
)

NUM_CHANNELS = 3


class Detector:  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """Object detection class using YOLOX to predict object bboxes.

    Attributes:
        logger (logging.Logger): Events logger.
        config (Dict[str, Any]): YOLOX node configuration.
        model_dir (pathlib.Path): Path to directory of model weights files.
        device (torch.device): Represents the device on which the torch.Tensor
            will be allocated.
        half (bool): Flag to determine if half-precision should be used.
        yolox (YOLOX): The YOLOX model for performing inference.
    """

    def __init__(
        self, config: Dict[str, Any], model_dir: Path, class_names: List[str]
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_dir = model_dir
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Half-precision only supported on CUDA
        self.config["half"] = self.config["half"] and self.device.type == "cuda"
        self.yolox = self._create_yolox_model()

    @torch.no_grad()
    def predict_object_bbox_from_image(
        self, image: np.ndarray, detect_ids: List[int]
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Detects bounding boxes of selected object categories from an image.

        The input image is first scaled according to the `input_size`
        configuration option. Detection results will be filtered according to
        `iou_threshold`, `score_threshold`, and `detect_ids` configuration
        options. Bounding boxes coordinates are then normalized w.r.t. the
        input `image` size.

        Args:
            image (np.ndarray): Input image.
            detect_ids (List[int]): List of label IDs to be detected

        Returns:
            (Tuple[List[np.ndarray], List[str], List[float]]): Returned tuple
                contains:
                - A list of detection bboxes
                - A list of human-friendly detection class names
                - A list of detection scores
        """
        self.update_detect_ids(detect_ids)
        # Store the original image size to normalize bbox later
        image_size = image.shape[:2]
        image, scale = self._preprocess(image)
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        image = image.half() if self.config["half"] else image.float()

        prediction = self.yolox(image)[0]
        bboxes, classes, scores = self._postprocess(
            prediction, scale, image_size, self.class_names
        )

        return bboxes, classes, scores

    def update_detect_ids(self, ids: List[int]) -> None:
        """Updates list of selected object category IDs. When the list is
        empty, all available object category IDs are detected.

        Args:
            ids: List of selected object category IDs
        """
        self.detect_ids = torch.Tensor(ids).to(self.device)  # type: ignore
        if self.config["half"]:
            self.detect_ids = self.detect_ids.half()

    def _create_yolox_model(self) -> YOLOX:
        """Creates a YOLOX model and loads its weights.

        Creates `detect_ids` as a `torch.Tensor`. Sets up `input_size` to a
        square shape. Logs model configurations.

        Returns:
            (YOLOX): YOLOX model.
        """
        self.input_size = (self.config["input_size"], self.config["input_size"])
        model_type = self.config["model_type"]
        model_path = self.model_dir / self.config["weights"]["model_file"][model_type]
        model_size = self.config["model_size"][model_type]

        self.logger.info(
            "YOLOX model loaded with the following configs:\n\t"
            f"Model type: {self.config['model_type']}\n\t"
            f"Input resolution: {self.input_size}\n\t"
            f"IDs being detected: {self.config['detect_ids']}\n\t"
            f"IOU threshold: {self.config['iou_threshold']}\n\t"
            f"Score threshold: {self.config['score_threshold']}\n\t"
            f"Class agnostic NMS: {self.config['agnostic_nms']}\n\t"
            f"Half-precision floating-point: {self.config['half']}\n\t"
            f"Fuse convolution and batch normalization layers: {self.config['fuse']}"
        )
        return self._load_yolox_weights(model_path, model_size)

    def _get_model(self, model_size: Dict[str, float]) -> YOLOX:
        """Constructs YOLOX model based on parsed configuration.

        Args:
            model_size (Dict[str, float]): Depth and width of the model.

        Returns:
            (YOLOX): YOLOX model.
        """
        return YOLOX(
            self.config["num_classes"],
            model_size["depth"],
            model_size["width"],
        )

    def _load_yolox_weights(
        self, model_path: Path, model_settings: Dict[str, float]
    ) -> YOLOX:
        """Loads YOLOX model weights.

        Args:
            model_path (Path): Path to model weights file.
            model_settings (Dict[str, float]): Depth and width of the model.

        Returns:
            (YOLOX): YOLOX model.

        Raises:
            ValueError: `model_path` does not exist.
        """
        if model_path.is_file():
            ckpt = torch.load(str(model_path), map_location="cpu")
            model = self._get_model(model_settings).to(self.device)
            if self.config["half"]:
                model.half()
            model.eval()
            model.load_state_dict(ckpt["model"])

            if self.config["fuse"]:
                model = fuse_model(model)
            return model
        raise ValueError(
            f"Model file does not exist. Please check that {model_path} exists."
        )

    def _postprocess(
        self,
        prediction: torch.Tensor,
        scale: float,
        image_shape: Tuple[int, int],
        class_names: List[str],
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Postprocesses detection result to be compatible with other nodes.

        Args:
            prediction (torch.Tensor): Raw detections from the YOLOX model.
            scale (float): Scale factor used during preprocessing.
            image_shape (Tuple[int, int]): Image size of the original input
                image.
            class_names (List[str]): List of human-friendly object category
                names.

        Returns:
            (Tuple[List[np.ndarray], List[str], List[float]]): Returned tuple
                contains:
                - A list of detection bboxes
                - A list of human-friendly detection class names
                - A list of detection scores
        """
        prediction[:, :4] = xywh2xyxy(prediction[:, :4])
        # Get score and class with highest confidence
        class_score, class_pred = torch.max(
            prediction[:, 5 : 5 + self.config["num_classes"]], 1, keepdim=True
        )
        # Filter by score_threshold
        conf_mask = (
            prediction[:, 4] * class_score.squeeze() >= self.config["score_threshold"]
        ).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((prediction[:, :5], class_score, class_pred.float()), 1)
        detections = detections[conf_mask]
        # Early return if all are below score_threshold
        if not detections.size(0):
            return np.empty((0, 4)), np.empty(0), np.empty(0)

        # Class agnostic NMS
        if self.config["agnostic_nms"]:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                self.config["iou_threshold"],
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                self.config["iou_threshold"],
            )
        output = detections[nms_out_index]

        # Filter by detect ids
        if self.detect_ids.size(0):
            output = output[torch.isin(output[:, 6], self.detect_ids)]
        output_np = output.cpu().detach().numpy()
        bboxes = xyxy2xyxyn(output_np[:, :4] / scale, *image_shape)
        scores = output_np[:, 4] * output_np[:, 5]
        classes = np.array([class_names[int(i)] for i in output_np[:, 6]])

        return bboxes, classes, scores

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocesses the input image for inference.

        The input image is resized to fit the `input_size` specified in the
        configurations. If the image cannot be resized exactly, a gray canvas,
        rgb(114, 114, 114), is created and the resized image is pasted on the
        top left corner of the canvas. rgb(114, 114, 114) is chosen based on
        the value used by the original repository during training.

        Args:
            image (np.ndarray): Input image.

        Returns:
            (Tuple[np.ndarray, float]): A tuple containing the preprocessed
                image and the scale factor used to resize the image. The shape
                of the preprocessed image is (C, H, W).
        """
        # Initialize canvas for padded image as gray
        padded_img = np.full(
            (self.input_size[0], self.input_size[1], NUM_CHANNELS), 114, dtype=np.uint8
        )
        scale = min(
            self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1]
        )
        scaled_height = int(image.shape[0] * scale)
        scaled_width = int(image.shape[1] * scale)
        resized_img = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[:scaled_height, :scaled_width] = resized_img

        # Rearrange from (H, W, C) to (C, H, W)
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, scale
