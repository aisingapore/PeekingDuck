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
Crowd counting class using csrnet model to predict density map and crowd count
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, Tuple, Callable

import cv2
import numpy as np
import tensorflow as tf


class Predictor:  # pylint: disable=too-few-public-methods
    """Crowd counting class using csrnet model to predict density map and crowd count"""

    def __init__(self, config: Dict[str, Any], model_dir: Path) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.model_dir = model_dir

        self.csrnet = self._create_csrnet_model()

    def _create_csrnet_model(self) -> Callable:
        """
        Creates csrnet model to predict density map and crowd count
        """
        model_type = self.config["model_type"]
        model_path = self.model_dir / self.config["weights"]["model_file"][model_type]

        self.logger.info(
            "CSRNet model loaded with following configs: \n\t"
            f"Model type: {self.config['model_type']}, \n\t"
            f"Input width: {self.config['width']} \n\t"
        )

        return self._load_csrnet(model_path)

    def _load_csrnet(self, model_path: Path) -> Callable:
        self.module = tf.saved_model.load(str(model_path))
        self.saved_model = self.module.signatures["serving_default"]

        return self.saved_model

    def predict_count_from_image(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Predicts density map and crowd count from image.

        Args:
            image (np.ndarray): input image.

        Returns:
            density_map (np.ndarray): resized density map.
            crowd_count (int): predicted count of people.
        """
        # 1. resizes and normalizes input image
        processed_image = self._process_input(image)

        # 2. generates the predicted density map
        density_map = self.saved_model(processed_image)["y_out"].numpy()

        # 3. resizes density map and counts the number of people
        density_map, crowd_count = self._process_output(image, density_map)

        return density_map, crowd_count

    def _process_input(self, image: np.ndarray) -> tf.Tensor:
        """Resizes and normalizes an image based on the mean and standard deviation
        of Imagenet. These are the default values for models with PyTorch origins.

        Args:
            image (np.ndarray): input image.

        Returns:
            image (np.ndarray): processed image.
        """
        image = self._resize_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        image = np.expand_dims(image, axis=0)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        return image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resizes an image based on the input width.

        Args:
            image (np.ndarray): input image.

        Returns:
            image (np.ndarray): resized image.
        """
        ratio = self.config["width"] / image.shape[1]
        dim = (self.config["width"], int(image.shape[0] * ratio))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def _process_output(image: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """Counts the number of people and resizes the output density map to match
        the original image size. The CSRNet model returns a density map that is
        1/8 the original image size. The resized density map is used to superimpose
        a heatmap over the original image.

        Args:
            density_map (np.ndarray): predicted density map.
            image (np.ndarray): input image.

        Returns:
            density_map (np.ndarray): resized density map.
            crowd_count (int): predicted count of people.
        """
        crowd_count = math.ceil(np.sum(density_map))

        density_map = density_map[0, :, :, 0]
        density_map = cv2.resize(
            density_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        return density_map, crowd_count
