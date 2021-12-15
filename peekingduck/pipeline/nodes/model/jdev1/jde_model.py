# Modifications copyright 2021 AI Singapore
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
# Original copyright (c) 2019 ZhongdaoWang
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

"""
JDE model for human detection and tracking.
"""

from typing import Any, Dict, List, Tuple
import logging
import numpy as np
import cv2
import torch
from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils.utils import letterbox
from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils.parse_config import (
    parse_model_cfg,
)
from peekingduck.pipeline.nodes.model.jdev1.jde_files.multitracker import JDETracker
from peekingduck.weights_utils import checker, downloader, finder


class JDEModel:  # pylint: disable=too-few-public-methods
    """JDE model that detects and tracks humans."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        # Check threshold values
        if not 0 <= config["iou_threshold"] <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0 <= config["score_threshold"] <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0 <= config["nms_threshold"] <= 1:
            raise ValueError("nms_threshold must be in [0, 1]")
        self.config = config

        # Check for JDE weights
        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )
        # pylint: disable=logging-fstring-interpolation
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("No weights detected. Proceeding to download...")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"Weights downloaded to {weights_dir}.")

        self.config["cfg"] = model_dir / config["weights"]["config_file"]
        self.config["weights_path"] = str(
            model_dir / config["weights"]["model_file"]["jde"]
        )
        cfg_dict = parse_model_cfg(self.config["cfg"])
        self.config["img_size"] = [
            int(cfg_dict[0]["width"]),
            int(cfg_dict[0]["height"]),
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker = JDETracker(self.config, self.device)

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Predicts person bboxes, object_id and labels.

        Args:
            frame (np.ndarray): image in numpy array.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Tuple containing
                detected bounding boxes, object tracking id and "person"
                label for each bounding box respectively.
        """
        img, img0 = self._reshape_image(frame)

        # Run detection and tracking
        blob = torch.from_numpy(img).unsqueeze(0).to(self.device)
        online_targets = self.tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for track in online_targets:
            tlwh = track.tlwh
            tid = track.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.config["min_box_area"] and not vertical:
                online_tlwhs.append(self._normalize_bboxes(tlwh))
                online_ids.append(str(tid))

        output_len = len(online_tlwhs)
        labels = ["person"] * output_len
        return online_tlwhs, online_ids, labels

    # pylint: disable=attribute-defined-outside-init
    def _reshape_image(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resizes image to the specified size in model config file.

        Args:
            frame (np.ndarray): Frame to reshape.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Normalized, and resized image.
        """
        img_width = self.config["img_size"][0]
        img_height = self.config["img_size"][1]

        frame_height, frame_width, _ = frame.shape
        self.width, self.height = self._get_size(
            frame_width, frame_height, img_width, img_height
        )
        img0 = cv2.resize(frame, (self.width, self.height))
        # Padded resize
        img, _, _, _ = letterbox(img0, height=img_height, width=img_width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img, img0

    @staticmethod
    def _get_size(
        frame_width: int, frame_height: int, img_width: int, img_height: int
    ) -> Tuple[int, int]:
        """Returns the requested size of image.

        Args:
            frame_width (int): Width of frame.
            frame_height (int): Height of frame.
            img_width (int): Image size width.
            img_height (int): Image size height.

        Returns:
            Tuple[int, int]: The requested size in pixels, as a 2-tuple:
                (width, height).
        """
        width_ratio = float(img_width) / frame_width
        height_ratio = float(img_height) / frame_height
        aspect_ratio = min(width_ratio, height_ratio)
        return int(frame_width * aspect_ratio), int(frame_height * aspect_ratio)

    def _normalize_bboxes(self, tlwh: np.ndarray) -> np.ndarray:
        """Normalizes coordinates of a bounding box. Divides x-coordinates
        by image width and y-coordinates by image height.

        Args:
            tlwh (np.ndarray): Bounding box in format
                `(top left x, top left y, width, height)`.

        Returns:
            np.ndarray: Normalized bounding box in format
                `(top left x, top left y, bottom right x, bottom right y)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        tlx, tly, blx, bly = ret

        n_tlx = tlx / self.width
        n_tly = tly / self.height
        n_blx = blx / self.width
        n_bly = bly / self.height

        return np.array([n_tlx, n_tly, n_blx, n_bly])
