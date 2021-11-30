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
    def __init__(self, config: Dict[str, Any], cuda_ctx=None) -> None:

        self.config = config
        self.root_dit = config["root"]
        self.logger = logging.getLogger(__name__)
        self.model_type = config["model_type"]

        # TF-TRT
        TRT_model_saved_dir = os.path.join(
            self.config["root"], self.config["converted_model_path"][self.model_type]
        )
        if not os.path.isdir(TRT_model_saved_dir):
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
        # input_fn: a generator function that yields input data as a list or tuple,
        # which will be used to execute the converted signature to generate TensorRT
        # engines.
        input_shape = (1, self.config["size"], self.config["size"], 3)
        batched_input = np.zeros(input_shape, dtype=np.float32)
        batched_input = tf.constant(batched_input)
        yield (batched_input,)

    def build_engine(self):

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
        t = time.process_time()
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

        elapsed_time = time.process_time() - t
        print(f"TensorRT engine built in {elapsed_time} secs")

    @staticmethod
    def bbox_scaling(bboxes: List[list], scale_factor: float) -> List[list]:
        """
        To scale the width and height of bboxes from v4tiny
        After the conversion of the model in .cfg and .weight file format, from
        Alexey's Darknet repo, to tf model, bboxes are bigger.
        So downscaling is required for a better fit
        """
        for idx, box in enumerate(bboxes):
            x_1, y_1, x_2, y_2 = tuple(box)
            center_x = (x_1 + x_2) / 2
            center_y = (y_1 + y_2) / 2
            scaled_x_1 = center_x - ((x_2 - x_1) / 2 * scale_factor)
            scaled_x_2 = center_x + ((x_2 - x_1) / 2 * scale_factor)
            scaled_y_1 = center_y - ((y_2 - y_1) / 2 * scale_factor)
            scaled_y_2 = center_y + ((y_2 - y_1) / 2 * scale_factor)
            bboxes[idx] = [scaled_x_1, scaled_y_1, scaled_x_2, scaled_y_2]

        return bboxes

    def predict(self, frame: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # TF-TRT
        image_data = cv2.resize(frame, (self.config["size"], self.config["size"]))
        image_data = image_data / 255.0
        image_data = np.asarray([image_data]).astype(np.float32)
        batch_data = tf.constant(image_data)
        func = self.model.signatures["serving_default"]
        output = func(batch_data)
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
        classes = classes.numpy()[0]
        classes = classes[: nums[0]]
        bboxes = bboxes.numpy()[0]
        bboxes = bboxes[: nums[0]]
        scores = scores.numpy()[0]
        scores = scores[: nums[0]]

        bboxes[:, [0, 1]] = bboxes[:, [1, 0]]  # swapping x and y axes
        bboxes[:, [2, 3]] = bboxes[:, [3, 2]]

        # scaling of bboxes if v4tiny model is used
        if self.config["model_type"] == "v4tiny":
            bboxes = self.bbox_scaling(bboxes, 0.75)

        return bboxes, classes, scores
