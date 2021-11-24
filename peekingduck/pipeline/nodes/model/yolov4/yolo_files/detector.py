# Copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Object detection class using yolo model to find object bboxes
"""

import builtins
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from peekingduck.pipeline.nodes.model.yolov4.yolo_files.dataset import transform_images
from peekingduck.utils.graph_functions import load_graph


class Detector:
    """Object detection class using yolo model to find object bboxes"""

    def __init__(self, config: Dict[str, Any], model_dir: Path) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.model_dir = model_dir

        self.yolo = self._create_yolo_model()

    def _create_yolo_model(self) -> tf.keras.Model:
        """
        Creates yolo model for human detection
        """
        model_type = self.config["model_type"]
        model_path = self.model_dir / self.config["weights"]["model_file"][model_type]

        self.logger.info(
            "Yolo model loaded with following configs: \n\t"
            f"Model type: {self.config['model_type']}, \n\t"
            f"Input resolution: {self.config['size']}, \n\t"
            f"IDs being detected: {self.config['detect_ids']} \n\t"
            f"Max Detections per class: {self.config['max_output_size_per_class']}, \n\t"
            f"Max Total Detections: {self.config['max_total_size']}, \n\t"
            f"IOU threshold: {self.config['yolo_iou_threshold']}, \n\t"
            f"Score threshold: {self.config['yolo_score_threshold']}"
        )

        return self._load_yolo_graph(model_path)

    def _load_yolo_graph(self, model_path: Path) -> tf.compat.v1.GraphDef:
        """
        When loading a graph model, you need to explicitly state the input
        and output nodes of the graph. It is usually x:0 for input and Identity:0
        for outputs, depending on how many output nodes you have.
        """
        model_type = f"yolo{self.config['model_type'][:2]}"
        model_nodes = self.config["MODEL_NODES"][model_type]
        if model_path.is_file():
            return load_graph(
                str(model_path),
                inputs=model_nodes["inputs"],
                outputs=model_nodes["outputs"],
            )
        raise ValueError(
            f"Graph file does not exist. Please check that {model_path} exists"
        )

    def _load_image(self, image_file: str) -> builtins.bytes:
        img = open(image_file, "rb").read()
        self.logger.info(f"image file {image_file} loaded")
        return img

    @staticmethod
    def _reshape_image(image: tf.Tensor, image_size: int) -> tf.Tensor:
        image = tf.expand_dims(image, 0)
        image = transform_images(image, image_size)
        return image

    @staticmethod
    def _shrink_dimension_and_length(
        boxes: tf.Tensor,
        scores: tf.Tensor,
        classes: tf.Tensor,
        nums: List[int],
        object_ids: List[int],
    ) -> Tuple[List[np.ndarray], List[float], List[str]]:
        len0 = nums[0]

        classes = classes.numpy()[0]
        classes = classes[:len0]
        mask1 = np.isin(
            classes, tuple(object_ids)
        )  # only identify objects we are interested in
        classes = tf.boolean_mask(classes, mask1)

        scores = scores.numpy()[0]
        scores = scores[:len0]
        scores = scores[mask1]

        boxes = boxes.numpy()[0]
        boxes = boxes[:len0]
        boxes = boxes[mask1]

        return boxes, scores, classes

    def _evaluate_image_by_yolo(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[float], List[int]]:
        """
        Takes in the yolo model and image to perform inference with.
        It will return the following:
            - boxes: the bounding boxes for each object
            - scores: the scores for each object predicted
            - classes: the class predicted for each bounding box
            - nums: number of valid bboxes. Only nums[0] should be used. The rest
                    are paddings.
        """
        # image = image[..., ::-1]  # swap from bgr to rgb
        pred = self.yolo(image)[-1]
        bboxes = pred[:, :, :4].numpy()
        bboxes[:, :, [0, 1]] = bboxes[:, :, [1, 0]]  # swapping x and y axes
        bboxes[:, :, [2, 3]] = bboxes[:, :, [3, 2]]
        pred_conf = pred[:, :, 4:]

        # performs nms using model's predictions
        boxes, scores, classes, nums = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bboxes, (tf.shape(bboxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            max_output_size_per_class=self.config["max_output_size_per_class"],
            max_total_size=self.config["max_total_size"],
            iou_threshold=self.config["yolo_iou_threshold"],
            score_threshold=self.config["yolo_score_threshold"],
        )
        return boxes, scores, classes, nums

    @staticmethod
    def _prepare_image_from_camera(image: np.ndarray) -> tf.Tensor:
        image = image.astype(np.float32)
        image = tf.convert_to_tensor(image)
        return image

    @staticmethod
    def _prepare_image_from_file(image: np.ndarray) -> tf.Tensor:
        image = tf.image.decode_image(image, channels=3)
        return image

    # possible that we may want to control what is being detection
    def predict_object_bbox_from_image(
        self, class_names: List[str], image: np.ndarray, detect_ids: List[int]
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Detect all objects' bounding box from one image

        Args:
            yolo (Model): model like yolov3 or yolov3_tiny
            image (np.array): input image

        Return:
            boxes (np.array): an array of bounding box with definition like
                (x1, y1, x2, y2), in a coordinate system with original point in
                the left top corner
        """
        # 1. prepare image
        image = self._prepare_image_from_camera(image)
        image = self._reshape_image(image, self.config["size"])

        # 2. evaluate image
        boxes, scores, classes, nums = self._evaluate_image_by_yolo(image)

        # 3. clean up return
        boxes, scores, classes = self._shrink_dimension_and_length(  # type: ignore
            boxes, scores, classes, nums, detect_ids
        )

        # convert classes into class names
        classes = np.array([class_names[int(i)] for i in classes])  # type: ignore

        return boxes, classes, scores  # type: ignore

    def setup_gpu(self) -> None:
        """Method to give info on whether the current device code is running on
        Is using GPU or CPU.
        """
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            self.logger.info(f"GPU setup with {len(physical_devices)} devices")
        else:
            self.logger.info("use CPU")
