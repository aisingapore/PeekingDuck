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
Detector class to handle detection of poses for HRNet
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.postprocessing import (
    affine_transform_xy,
    get_keypoint_conns,
    get_valid_keypoints,
    reshape_heatmaps,
    scale_transform,
)
from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.preprocessing import (
    crop_and_resize,
    tlwh2xywh,
)
from peekingduck.pipeline.utils.bbox.transforms import xyxyn2tlwh
from peekingduck.utils.graph_functions import load_graph


class Detector:  # pylint: disable=too-few-public-methods
    """Detector class to handle detection of poses for HRNet."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        model_type: str,
        model_file: Dict[str, str],
        model_nodes: Dict[str, List[str]],
        resolution: Dict[str, int],
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.model_type = model_type
        self.model_path = model_dir / model_file[self.model_type]
        self.model_nodes = model_nodes
        self.resolution = resolution
        self.score_threshold = score_threshold

        self.hrnet = self._create_hrnet_model()

    def predict(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """HRnet prediction function.

        Args:
            frame (np.ndarray): Image in numpy array.
            bboxes (np.ndarray): Array of detected bboxes.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing list of
            bboxes and pose related info, i.e., coordinates, scores, and
            connections
        """
        cropped_frames, affine_matrices, frame_size = self._preprocess(frame, bboxes)
        heatmaps = self.hrnet(tf.cast(cropped_frames, float))[0]

        cropped_frames_scale = [cropped_frames.shape[2], cropped_frames.shape[1]]
        poses, keypoint_scores, keypoint_conns = self._postprocess(
            heatmaps.numpy(), affine_matrices, cropped_frames_scale, frame_size
        )

        return poses, keypoint_scores, keypoint_conns

    def _create_hrnet_model(self) -> Callable:
        resolution_tuple = (self.resolution["height"], self.resolution["width"])
        self.logger.info(
            "HRNet graph model loaded with following configs:\n\t"
            f"Resolution: {resolution_tuple},\n\t"
            f"Score threshold: {self.score_threshold}"
        )
        return self._load_hrnet_weights()

    def _load_hrnet_weights(self) -> Callable:
        return load_graph(
            str(self.model_path),
            inputs=self.model_nodes["inputs"],
            outputs=self.model_nodes["outputs"],
        )

    def _postprocess(  # pylint: disable=too-many-locals
        self,
        heatmaps: np.ndarray,
        affine_matrices: np.ndarray,
        cropped_frames_scale: List[int],
        frame_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post processes output heatmaps to required keypoint arrays.

        Args:
            heatmaps (np.ndarray): Output heatmaps from hrnet network.
            affine_matrices (np.ndarray): transformation matrices of preprocess
                cropping.
            cropped_frames_scale (List[int]): Shape of cropped bboxes.
            frame_size (Tuple[int, int]): Size of original image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple containing array
            of bboxes and pose related info, i.e., coordinates, scores, and
            connections
        """
        batch, out_h, out_w, num_joints = heatmaps.shape
        heatmaps_reshaped = reshape_heatmaps(heatmaps)

        max_idxs = np.argmax(heatmaps_reshaped, 2)
        kp_scores = np.amax(heatmaps_reshaped, 2)
        keypoints = (
            np.repeat(max_idxs, 2).reshape(batch, num_joints, -1).astype(np.float32)
        )
        keypoints[:, :, 0] = (keypoints[:, :, 0]) % out_w
        keypoints[:, :, 1] = np.floor((keypoints[:, :, 1]) / out_w)

        keypoints = scale_transform(
            keypoints, in_scale=[out_w, out_h], out_scale=cropped_frames_scale
        )
        keypoints = affine_transform_xy(keypoints, affine_matrices)
        keypoints, kp_masks = get_valid_keypoints(
            keypoints, kp_scores, batch, self.score_threshold
        )

        normalized_keypoints = keypoints / frame_size
        keypoint_conns = get_keypoint_conns(normalized_keypoints, kp_masks)

        return normalized_keypoints, kp_scores, keypoint_conns

    def _preprocess(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Crops bboxes while preserving aspect ratio.

        Args:
            frame (np.ndarray): Input image in numpy array.
            bboxes (np.ndarray): Array of detected bboxes.

        Returns:
            (Tuple[np.ndarray, np.ndarray, Tuple[int, int]]): Array of cropped
            bboxes, transformation matrices, and original frame size.
        """
        frame = frame / 255.0
        frame_size = (frame.shape[1], frame.shape[0])
        cropped_size = (self.resolution["width"], self.resolution["height"])

        tlwhs = xyxyn2tlwh(bboxes, frame.shape[0] - 1, frame.shape[1] - 1)
        xywhs = tlwh2xywh(tlwhs, self.resolution["width"] / self.resolution["height"])
        cropped_imgs, affine_matrices = crop_and_resize(frame, xywhs, cropped_size)

        return np.array(cropped_imgs), affine_matrices, frame_size
