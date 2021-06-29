# Copyright 2021 AI Singapore
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
Detector class to handle detection of poses for hrnet
"""

import os
import logging
from typing import Any, Dict, List, Tuple, Callable

import tensorflow as tf
import numpy as np
from peekingduck.utils.graph_functions import load_graph
from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.preprocessing \
    import project_bbox, box2cs, crop_and_resize
from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.postprocessing \
    import scale_transform, affine_transform_xy, reshape_heatmaps, \
    get_valid_keypoints, get_keypoint_conns


class Detector:
    """Detector class to handle detection of poses for hrnet
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config['root']
        self.resolution = config['resolution']
        self.min_score = config['score_threshold']

        self.hrnet = self._create_hrnet_model()

    def _inference_function(self, person_frame: np.ndarray, training: bool = False) -> tf.Tensor:  # pylint:disable=unused-argument
        '''
        When graph is frozen, we need a different way to extract the
        arrays. We use this to return the values needed. The purpose
        of this function is to make it consistent with how our regular
        hrnet is used for inference.

        The "training" argument is needed because in the usual implementation
        of hrnet has the training argument. It does nothing here.
        '''
        heatmap = self.frozen_fn(tf.cast(person_frame, float))[0]
        return heatmap

    def _create_hrnet_model(self) -> Callable:
        graph_path = os.path.join(self.root_dir, self.config['model_file'])
        model_nodes = self.config['MODEL_NODES']
        self.frozen_fn = load_graph(graph_path,
                                    inputs=model_nodes['inputs'],
                                    outputs=model_nodes['outputs'])
        resolution_tuple = (self.resolution['height'], self.resolution['width'])
        self.logger.info(
            'HRNet graph model loaded with following configs: \n \
            Resolution: %s, \n \
            Score Threshold: %s', resolution_tuple, self.min_score)
        return self._inference_function

    def preprocess(self,
                   frame: np.ndarray,
                   bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Preprocessing function that crops bboxes while preserving aspect ratio

        Args:
            frame (np.ndarray): input image in numpy array
            bboxes (np.ndarray): array of detected bboxes

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[int, int]]: array of cropped bboxes, \
            transformation matrices and original frame size
        """
        frame = frame / 255.0
        frame_size = (frame.shape[1], frame.shape[0])
        cropped_size = (self.resolution['width'], self.resolution['height'])

        projected_bbox = project_bbox(bboxes, frame_size)
        center_bbox = box2cs(projected_bbox, self.resolution['width'] / self.resolution['height'])
        cropped_imgs, affine_matrices = crop_and_resize(frame, center_bbox, cropped_size)

        return np.array(cropped_imgs), affine_matrices, frame_size

    def postprocess(self, heatmaps: np.ndarray,  # pylint: disable=too-many-locals
                    affine_matrices: np.ndarray,
                    cropped_frames_scale: List[int],
                    frame_size: Tuple[int, int]) -> Tuple[np.ndarray,
                                                          np.ndarray,
                                                          np.ndarray]:
        """Postprocessing function to process output heatmaps to required keypoint arays

        Args:
            heatmaps (np.ndarray): Output heatmaps from hrnet network
            affine_matrices (np.ndarray): transformation matrices of preprocess cropping
            cropped_frames_scale (List[int]): Shape of cropped bboxes
            frame_size (Tuple[int, int]): Size of original image

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: \
            tuple containing array of bboxes and pose related info i.e coordinates,
            scores, connections
        """
        batch, out_h, out_w, num_joints = heatmaps.shape
        heatmaps_reshaped = reshape_heatmaps(heatmaps)

        max_idxs = np.argmax(heatmaps_reshaped, 2)
        kp_scores = np.amax(heatmaps_reshaped, 2)
        keypoints = np.repeat(max_idxs, 2).reshape(batch, num_joints, -1).astype(np.float32)
        keypoints[:, :, 0] = (keypoints[:, :, 0]) % out_w
        keypoints[:, :, 1] = np.floor((keypoints[:, :, 1]) / out_w)

        keypoints = scale_transform(keypoints,
                                    in_scale=[out_w, out_h],
                                    out_scale=cropped_frames_scale)
        keypoints = affine_transform_xy(keypoints, affine_matrices)
        keypoints, kp_masks = get_valid_keypoints(
            keypoints, kp_scores, batch, self.min_score)

        normalized_keypoints = keypoints / frame_size

        keypoint_conns = get_keypoint_conns(normalized_keypoints, kp_masks)

        return normalized_keypoints, kp_scores, keypoint_conns

    def predict(self, frame: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray,
                                                                      np.ndarray,
                                                                      np.ndarray]:
        """HRnet prediction function

        Args:
            frame (np.ndarray): Image in numpy array
            bboxes (np.ndarray): Array of detected bboxes

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: \
            tuple containing list of bboxes and pose related info i.e coordinates,
            scores, connections
        """
        cropped_frames, affine_matrices, frame_size = self.preprocess(frame, bboxes)
        heatmaps = self.hrnet(cropped_frames, training=False)

        cropped_frames_scale = [cropped_frames.shape[2], cropped_frames.shape[1]]
        poses, kp_scores, kp_conns = self.postprocess(heatmaps.numpy(), affine_matrices,
                                                      cropped_frames_scale, frame_size)

        return poses, kp_scores, kp_conns
