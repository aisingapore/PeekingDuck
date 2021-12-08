# Modifications copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original copyright (c) 2019 ZhongdaoWang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
JDE model for human detection and tracking
"""

from typing import Any, Dict, List, Tuple
import logging
import numpy as np
import cv2
import torch
from peekingduck.pipeline.nodes.model.jde_mot.jde_files.utils.utils import letterbox
from peekingduck.pipeline.nodes.model.jde_mot.jde_files.utils.parse_config import (
    parse_model_cfg,
)
from peekingduck.pipeline.nodes.model.jde_mot.jde_files.multitracker import JDETracker
from peekingduck.weights_utils import checker, downloader, finder

# pylint: disable=invalid-name, no-member, no-self-use, attribute-defined-outside-init, logging-fstring-interpolation, too-few-public-methods


class JDE:
    """JDE model"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Check threshold values
        if not 0 <= config["iou_threshold"] <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0 <= config["conf_threshold"] <= 1:
            raise ValueError("conf_threshold must be in [0, 1]")
        if not 0 <= config["nms_threshold"] <= 1:
            raise ValueError("nms_threshold must be in [0, 1]")
        self.config = config

        # Check for JDE weights
        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("No weights detected. Proceeding to download...")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"Weights downloaded to {weights_dir}.")

        self.config["cfg"] = str(model_dir / config["weights"]["config_file"])
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
            frame (np.ndarray): image in numpy array

        Returns:
            bboxes (np.ndarray): numpy array of detected bboxes
            obj_id (np.ndarray): numpy array of object tracking id
            label (list): list of label of "person" for bboxes
        """
        img, img0 = self._reshape_image(frame)

        # Run detection and tracking
        blob = torch.from_numpy(img).unsqueeze(0).to(self.device)
        online_targets = self.tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.config["min_box_area"] and not vertical:
                online_tlwhs.append(self._normalize_bboxes(tlwh))
                online_ids.append(str(tid))

        output_len = len(online_tlwhs)
        labels = ["person"] * output_len
        return online_tlwhs, online_ids, labels

    def _reshape_image(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape imput frame.

        Args:
            frame (np.ndarray): Frame to reshape

        Returns:
            Tuple[np.ndarray, np.ndarray]: Normalized, resized frames.
        """
        width = self.config["img_size"][0]
        height = self.config["img_size"][1]

        vh, vw, _ = frame.shape
        self.w, self.h = self._get_size(vw, vh, width, height)
        img0 = cv2.resize(frame, (self.w, self.h))
        # Padded resize
        img, _, _, _ = letterbox(img0, height=height, width=width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img, img0

    def _get_size(self, vw: int, vh: int, dw: int, dh: int) -> Tuple[int, int]:
        """Get size of frame.

        Args:
            vw (int): Frame width.
            vh (int): Frame height.
            dw (int): Image size width.
            dh (int): Image size height.

        Returns:
            Tuple[int, int]: Size of frame.
        """
        wa = float(dw) / vw
        ha = float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def _normalize_bboxes(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding boxes from tlwh to normalized tlbr.

        Args:
            tlwh (np.ndarray): Bounding box in format
                `(top left x, top left y, width, height)`

        Returns:
            np.ndarray: Normalized bounding box in format
                `(top left x, top left y, bottom right x, bottom right y)`
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        tlx, tly, blx, bly = ret

        n_tlx = tlx / self.w
        n_tly = tly / self.h
        n_blx = blx / self.w
        n_bly = bly / self.h

        return np.array([n_tlx, n_tly, n_blx, n_bly])
