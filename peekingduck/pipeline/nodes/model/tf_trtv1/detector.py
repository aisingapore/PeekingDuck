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
Object detection class using TF_TRT yolo single label model
to find license plate object bboxes
"""

import time
import os
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.saved_model import tag_constants


class Detector:
    """
    Object detection class using TF_TRT yolo model to find object bboxes
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.root_dit = config["root"]
        self.logger = logging.getLogger(__name__)
        self.model_type = config["model_type"]

        # TF-TRT
        trt_model_saved_dir = os.path.join(
            self.config["root"], self.config["converted_model_path"][self.model_type]
        )
        if not os.path.isdir(trt_model_saved_dir):
            self.build_engine()

        self.model = self._create_yolo_model()

    def _create_yolo_model(self) -> Any:
        """
        Creates yolo model for license plate detection
        """

        # Load converted model and infer
        converted_model_path = os.path.join(
            self.config["root"], self.config["converted_model_path"][self.model_type]
        )
        model = tf.saved_model.load(converted_model_path, tags=[tag_constants.SERVING])

        return model

    def my_input_fn(self):
        """
        generator function that yields input data as a list or tuple,
        which will be used to execute the converted signature to generate TensorRT
        engines
        """
        input_shape = (1, self.config["size"], self.config["size"], 3)
        batched_input = np.zeros(input_shape, dtype=np.float32)
        batched_input = tf.constant(batched_input)
        yield (batched_input,)

    def build_engine(self):
        """
        Prebuild the tensorRT enginer prior to running inference
        """
        input_saved_model_dir = os.path.join(
            self.config["root"], self.config["model_weights_dir"][self.model_type]
        )

        # Conversion Parameters
        conversion_params = trt_convert.TrtConversionParams(
            precision_mode=trt_convert.TrtPrecisionMode.FP16,
            max_workspace_size_bytes=4000000000,
            max_batch_size=1,
        )

        converter = trt_convert.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params,
        )

        print("Building the TensorRT engine.  This would take a while...")
        curr_time = time.process_time()
        # Converter method used to partition and optimize TensorRT compatible segments
        converter.convert()

        # Optionally, build TensorRT engines before deployment to save time at runtime
        # Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime
        converter.build(input_fn=self.my_input_fn)

        # Save the converted model
        converted_model_path = os.path.join(
            self.config["root"], self.config["converted_model_path"][self.model_type]
        )
        converter.save(converted_model_path)

        elapsed_time = time.process_time() - curr_time
        print(f"TensorRT engine built in {elapsed_time} secs")

    @staticmethod
    def bbox_scaling(bboxes: List[list], scaling_factor: float) -> List[list]:
        """
        To scale the width and height of bboxes from v4tiny
        After the conversion of the model in .cfg and .weight file format, from
        Alexey's Darknet repo, to tf model, bboxes are bigger.
        So downscaling is required for a better fit
        """
        for idx, box in enumerate(bboxes):
            xmin, ymin, xmax, ymax = tuple(box)
            middle_y = (ymin + ymax) / 2
            middle_x = (xmin + xmax) / 2
            y1_scaled = middle_y - ((ymax - ymin) / 2 * scaling_factor)
            y2_scaled = middle_y + ((ymax - ymin) / 2 * scaling_factor)
            x1_scaled = middle_x - ((xmax - xmin) / 2 * scaling_factor)
            x2_scaled = middle_x + ((xmax - xmin) / 2 * scaling_factor)
            bboxes[idx] = [x1_scaled, y1_scaled, x2_scaled, y2_scaled]

        return bboxes

    def predict(self, frame: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect all license plate objects' bounding box from one image

        args:
                image: (Numpy Array) input image

        return:
                boxes: (Numpy Array) an array of bounding box with
                    definition like (x1, y1, x2, y2), in a
                    coordinate system with origin point in
                    the left top corner
                labels: (Numpy Array) an array of class labels
                scores: (Numpy Array) an array of confidence scores
        """
        # TF-TRT
        image_data = cv2.resize(frame, (self.config["size"], self.config["size"]))
        image_data = image_data / 255.0
        image_data = np.asarray([image_data]).astype(np.float32)
        image_data = tf.constant(image_data)
        func = self.model.signatures["serving_default"]
        output = func(image_data)
        for _, value in output.items():
            pred_conf = value[:, :, 4:]
            boxes = value[:, :, 0:4]

        # performs nms using model's predictions
        bboxes, scores, classes, nums = tf.image.combined_non_max_suppression(
            tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            self.config["max_output_size_per_class"],
            self.config["max_total_size"],
            self.config["yolo_iou_threshold"],
            self.config["yolo_score_threshold"],
        )
        class_labels = classes.numpy()[0]
        class_labels = class_labels[: nums[0]]
        bboxes = bboxes.numpy()[0]
        bboxes = bboxes[: nums[0]]
        bbox_scores = scores.numpy()[0]
        bbox_scores = bbox_scores[: nums[0]]
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]
        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]  # swapping x and y axes

        # scaling of bboxes if v4tiny model is used
        if self.config["model_type"] == "v4tiny":
            bboxes = self.bbox_scaling(bboxes, 0.75)

        return bboxes, class_labels, bbox_scores
