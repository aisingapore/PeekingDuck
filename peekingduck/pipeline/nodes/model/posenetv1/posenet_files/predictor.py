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
Predictor class to handle detection of poses for posenet
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.constants import (
    KEYPOINTS_NUM,
    MIN_PART_SCORE,
    SCALE_FACTOR,
    SKELETON,
)
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.detector import (
    detect_keypoints,
    get_keypoints_relative_coords,
)
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.preprocessing import (
    rescale_image,
)
from peekingduck.utils.graph_functions import load_graph

OUTPUT_STRIDE = 16


class Predictor:  # pylint: disable=too-many-instance-attributes
    """Predictor class to handle detection of poses for posenet"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        model_type: str,
        model_file: Dict[str, str],
        model_nodes: Dict[str, Dict[str, List[str]]],
        resolution: Dict[str, int],
        max_pose_detection: int,
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.model_type = model_type
        self.model_path = model_dir / model_file[self.model_type]
        self.model_nodes = model_nodes[
            "resnet" if self.model_type == "resnet" else "mobilenet"
        ]

        self.resolution = self.get_resolution_as_tuple(resolution)
        self.max_pose_detection = max_pose_detection
        self.score_threshold = score_threshold

        self.posenet = self._create_posenet_model()

    def predict(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # pylint: disable=too-many-locals
        """PoseNet prediction function

        Args:
            frame (np.array): image in numpy array

        Returns:
            bboxes (np.ndarray): array of bboxes
            keypoints (np.ndarray): array of keypoint coordinates
            keypoints_scores (np.ndarray): array of keypoint scores
            keypoints_conns (np.ndarray): array of keypoint connections
        """
        (
            full_keypoint_rel_coords,
            full_keypoint_scores,
            full_masks,
        ) = self._predict_all_poses(frame)

        bboxes = []
        keypoints = []
        keypoint_scores = []
        keypoint_conns = []

        for pose_coords, pose_scores, pose_masks in zip(
            full_keypoint_rel_coords, full_keypoint_scores, full_masks
        ):
            bbox = self._get_bbox_of_one_pose(pose_coords, pose_masks)
            pose_coords = self._get_valid_full_keypoints_coords(pose_coords, pose_masks)
            pose_connections = self._get_connections_of_one_pose(
                pose_coords, pose_masks
            )
            bboxes.append(bbox)
            keypoints.append(pose_coords)
            keypoint_scores.append(pose_scores)
            keypoint_conns.append(pose_connections)

        return (
            np.array(bboxes),
            np.array(keypoints),
            np.array(keypoint_scores),
            np.array(keypoint_conns),
        )

    def _create_posenet_model(self) -> Callable:
        self.logger.info(
            "PoseNet model loaded with following configs:\n\t"
            f"Model type: {self.model_type},\n\t"
            f"Input resolution: {self.resolution},\n\t"
            f"Max pose detection: {self.max_pose_detection},\n\t"
            f"Score threshold: {self.score_threshold}"
        )
        return self._load_posenet_weights()

    def _load_posenet_weights(self) -> Callable:
        if not self.model_path.is_file():
            raise ValueError(
                f"Graph file does not exist. Please check that {self.model_path} exists"
            )
        return load_graph(
            str(self.model_path),
            inputs=self.model_nodes["inputs"],
            outputs=self.model_nodes["outputs"],
        )

    def _predict_all_poses(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict relative coordinates, confident scores and validation masks
        for all detected poses

        Args:
            frame (np.ndarray): image for inference

        Returns:
            full_keypoint_coords (np.ndarray): keypoints coordinates of
                detected poses
            full_keypoint_scores (np.ndarray): keypoints confidence scores of
                detected poses
            full_masks (np.ndarray): keypoints validation masks of detected
                poses
        """
        image, output_scale, image_size = self._create_image_from_frame(
            OUTPUT_STRIDE, frame, self.resolution, self.model_type
        )

        dst_scores = np.zeros((self.max_pose_detection, KEYPOINTS_NUM))
        dst_keypoints = np.zeros((self.max_pose_detection, KEYPOINTS_NUM, 2))

        pose_count = detect_keypoints(
            self.posenet,
            image,
            OUTPUT_STRIDE,
            dst_scores,
            dst_keypoints,
            self.model_type,
            self.score_threshold,
        )
        full_keypoint_scores = dst_scores[:pose_count]
        full_keypoint_coords = dst_keypoints[:pose_count]

        full_keypoint_rel_coords = get_keypoints_relative_coords(
            full_keypoint_coords, output_scale, image_size
        )

        full_masks = self._get_full_masks_from_keypoint_scores(full_keypoint_scores)

        return full_keypoint_rel_coords, full_keypoint_scores, full_masks

    @staticmethod
    def get_resolution_as_tuple(resolution: Dict[str, int]) -> Tuple[int, int]:
        """Convert resolution from dict to tuple format

        Args:
            resolution (Dict[str, int]): height and width in dict format

        Returns:
            resolution (Tuple(int)): height and width in tuple format
        """
        return int(resolution["height"]), int(resolution["width"])

    @staticmethod
    def _create_image_from_frame(
        output_stride: int,
        frame: np.ndarray,
        input_res: Tuple[int, int],
        model_type: str,
    ) -> Tuple[tf.Tensor, np.ndarray, List[int]]:
        """Rescale raw frame and convert to tensor image for inference"""
        image_size = [frame.shape[1], frame.shape[0]]

        image, output_scale = rescale_image(
            frame,
            input_res,
            SCALE_FACTOR,
            output_stride,
            model_type,
        )
        image = tf.convert_to_tensor(image)
        return image, output_scale, image_size

    @staticmethod
    def _get_bbox_of_one_pose(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Get the bounding box bordering the keypoints of a single pose"""
        coords = coords[mask, :]
        if coords.shape[0]:
            min_x, min_y, max_x, max_y = (
                coords[:, 0].min(),
                coords[:, 1].min(),
                coords[:, 0].max(),
                coords[:, 1].max(),
            )
            bbox = [min_x, min_y, max_x, max_y]
            return np.array(bbox)
        return np.zeros(0)

    @staticmethod
    def _get_connections_of_one_pose(
        coords: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Get connections between adjacent keypoint pairs if both keypoints are
        detected
        """
        connections = []
        for start_joint, end_joint in SKELETON:
            if masks[start_joint - 1] and masks[end_joint - 1]:
                connections.append((coords[start_joint - 1], coords[end_joint - 1]))
        return np.array(connections)

    @staticmethod
    def _get_full_masks_from_keypoint_scores(keypoint_scores: np.ndarray) -> np.ndarray:
        """Obtain masks for keypoints with confidence scores above the detection
        threshold
        """
        masks = keypoint_scores > MIN_PART_SCORE
        return masks

    @staticmethod
    def _get_valid_full_keypoints_coords(
        coords: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Apply masks to keep only valid keypoints' relative coordinates

        Args:
            coords (np.array): Nx2 array of keypoints' relative coordinates
            masks (np.array): masks for valid (> min confidence score) keypoints

        Returns:
            full_joints (np.array): Nx2 array of keypoints where undetected
                keypoints are assigned a (-1) value.
        """
        full_joints = coords.copy()
        full_joints = np.clip(full_joints, 0, 1)
        full_joints[~masks] = -1
        return full_joints
