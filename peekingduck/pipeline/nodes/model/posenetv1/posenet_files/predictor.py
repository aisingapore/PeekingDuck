"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import logging
from typing import Dict, List, Callable, Any, Tuple
import tensorflow as tf
import numpy as np

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.posedata import PoseData
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.detector import detect_keypoints, \
    get_keypoints_relative_coords
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.constants import SCALE_FACTOR, \
    KEYPOINTS_NUM, MIN_PART_SCORE
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.preprocessing import rescale_image
from peekingduck.pipeline.nodes.model.utils.graph_functions import load_graph


class Predictor:  # pylint: disable=too-many-instance-attributes
    """Predictor class to handle detection of poses for posenet
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.model_type = self.config['model_type']

        self.posenet_model = self._create_posenet_model()

    def _create_posenet_model(self) -> tf.keras.Model:
        self.output_stride = self.config['output_stride']
        self.resolution = self.get_resolution_as_tuple(self.config['resolution'])
        self.max_pose_detection = self.config['max_pose_detection']
        self.score_threshold = self.config['score_threshold']

        model_path = os.path.join(
            self.config['root'], self.config['model_files'][self.model_type])
        model_func = self._load_posenet_graph(model_path)

        self.logger.info(
            'PoseNet model loaded with following configs: \n \
            Model type: %s, \n \
            Output stride: %s, \n \
            Input resolution: %s, \n \
            Max pose detection: %s, \n \
            Score threshold: %s', self.model_type, self.output_stride, self.resolution,
            self.max_pose_detection, self.score_threshold)

        return model_func

    def _load_posenet_graph(self, filepath: str) -> Callable:
        model_id = 'mobilenet'
        if self.model_type == 'resnet':
            model_id = 'resnet'
        model_nodes = self.config['MODEL_NODES'][model_id]
        model_path = os.path.join(filepath)
        if os.path.isfile(model_path):
            return load_graph(model_path, inputs=model_nodes['inputs'],
                              outputs=model_nodes['outputs'])
        raise ValueError('Graph file does not exist. Please check that '
                         '%s exists' % model_path)

    @staticmethod
    def get_resolution_as_tuple(resolution: dict) -> Tuple[int, int]:
        """ Convert resolution from dict to tuple format

        Args:
            resolution (dict): height and width in dict format

        Returns:
            resolution (Tuple(int)): height and width in tuple format
        """
        res1, res2 = resolution['height'], resolution['width']

        return (int(res1), int(res2))

    def predict(self,
                frame: np.ndarray) -> List[PoseData]:
        """ PoseNet prediction function

        Args:
            frame (np.array): image in numpy array

        Returns:
            poses (List[PoseData]): list of PoseData object
        """
        full_keypoint_rel_coords, full_keypoint_scores, full_masks = \
            self._predict_all_poses(
                self.posenet_model,
                frame,
                self.model_type)

        poses = []

        for coords, scores, masks in zip(full_keypoint_rel_coords,
                                         full_keypoint_scores, full_masks):
            pose = PoseData(keypoints=coords, keypoint_scores=scores,
                            masks=masks, connections=None)
            poses.append(pose)

        return poses

    def _predict_all_poses(
            self,
            posenet_model: tf.keras.Model,
            frame: np.ndarray,
            model_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict relative coordinates, confident scores and validation masks
        for all detected poses

        Args:
            posenet model: tensorflow model
            frame (np.array): image for inference
            model_type (str): specified model type (refer to modelconfig.yml)

        Returns:
            full_keypoint_coords (np.array): keypoints coordinates of detected poses
            full_keypoint_scores (np.array): keypoints confidence scores of detected
                poses
            full_masks (np.array): keypoints validation masks of detected poses
        """
        image, output_scale, image_size = self._create_image_from_frame(
            self.output_stride, frame, self.resolution, model_type)

        dst_scores = np.zeros(
            (self.max_pose_detection, KEYPOINTS_NUM))
        dst_keypoints = np.zeros(
            (self.max_pose_detection, KEYPOINTS_NUM, 2))

        pose_count = detect_keypoints(
            posenet_model, image, self.output_stride, dst_scores, dst_keypoints,
            model_type, self.score_threshold)
        full_keypoint_scores = dst_scores[:pose_count]
        full_keypoint_coords = dst_keypoints[:pose_count]

        full_keypoint_rel_coords = get_keypoints_relative_coords(
            full_keypoint_coords, output_scale, image_size)

        full_masks = self._get_full_masks_from_keypoint_scores(full_keypoint_scores)

        return full_keypoint_rel_coords, full_keypoint_scores, full_masks

    @staticmethod
    def _create_image_from_frame(output_stride: int,
                                 frame: np.ndarray,
                                 input_res: Tuple[int, int],
                                 model_type: str) -> Tuple[tf.Tensor, np.ndarray, List[int]]:
        """ Preprocess raw frame to image for inference

        Args:
            output_stride (int): output stride to convert output indices to image coordinates
            frame (np.array): image for inference
            input_res (int): input resolution to model
            model_type (str): specified model type (refer to modelconfig.yml)

        Returns:
            image (np.array): image for inference
            output_scale (np.array): output scale
            image_size (list): list of frame shape in height and width format
        """
        image_size = [frame.shape[1], frame.shape[0]]

        image, output_scale = rescale_image(frame,
                                            input_res,
                                            scale_factor=SCALE_FACTOR,
                                            output_stride=output_stride,
                                            model_type=model_type
                                            )
        image = tf.convert_to_tensor(image)
        return image, output_scale, image_size

    @staticmethod
    def _get_full_masks_from_keypoint_scores(keypoint_scores: np.ndarray) -> np.ndarray:
        """ Obtain masks for keypoints with confidence scores above the detection threshold
        """
        masks = keypoint_scores > MIN_PART_SCORE
        return masks
