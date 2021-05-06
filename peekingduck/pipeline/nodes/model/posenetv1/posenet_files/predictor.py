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
from typing import Dict, Any, List, Tuple
import tensorflow as tf
import numpy as np

from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.posedata import PoseData
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.detector import detect_keypoints, \
    get_keypoints_relative_coords
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.constants import SCALE_FACTOR, \
    KEYPOINTS_NUM, MIN_PART_SCORE
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.preprocessing import rescale_image
from peekingduck.pipeline.nodes.model.posenetv1.posenet_files.graph_functions import load_graph


class Predictor:
    """Predictor class to handle detection of poses for posenet
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config['root']

        self.output_stride = self.config['output_stride']
        self.resolution = self.get_resolution_as_tuple(self.config['resolution'])
        self.max_pose_detection = self.config['max_pose_detection']
        self.model_type = self.config['model_type']
        self.score_threshold = self.config['score_threshold']
        self.dst_scores = np.zeros(
            (self.max_pose_detection, KEYPOINTS_NUM))
        self.dst_keypoints = np.zeros(
            (self.max_pose_detection, KEYPOINTS_NUM, 2))

        self.posenet_model = self._create_posenet_model()

    def _create_posenet_model(self):
        # maybe should rename to graph_files instead of model_files
        model_type = self.config['model_type']
        model_path = os.path.join(self.root_dir, self.config['model_files'][model_type])
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

    def _load_posenet_graph(self, filepath):
        model_id = 'mobilenet'
        if self.config['model_type'] == 'resnet':
            model_id = 'resnet'
        model_nodes = self.config['MODEL_NODES'][model_id]
        model_path = os.path.join(filepath)
        if os.path.isfile(model_path):
            return load_graph(model_path, inputs=model_nodes['inputs'],
                              outputs=model_nodes['outputs'])
        raise ValueError('Graph file does not exist. Please check that '
                         '%s exists' % model_path)

    @staticmethod
    def get_resolution_as_tuple(resolution: dict):
        """ Convert resolution from dict to tuple format

        Args:
            resolution (dict): height and width in dict format

        Returns:
            resolution (Tuple(int)): height and width in tuple format
        """
        res1, res2 = resolution['height'], resolution['width']

        return (int(res1), int(res2))

    def predict(self,
                frame: List[List[float]]):
        """ PoseNet prediction function

        Args:
            frame (np.array): image in numpy array

        Returns:
            poses (List[PoseData]): list of PoseData object
        """
        full_keypoint_rel_coords, full_keypoint_scores, full_masks = \
            self._predict_all_poses(
                self.posenet_model,
                self.output_stride,
                frame,
                self.resolution,
                self.dst_scores,
                self.dst_keypoints,
                self.model_type,
                self.score_threshold)

        poses = []

        for coords, scores, masks in zip(full_keypoint_rel_coords,
                                         full_keypoint_scores, full_masks):
            pose = PoseData(keypoints=coords, keypoint_scores=scores,
                            masks=masks, connections=None)
            poses.append(pose)

        return poses

    def _predict_all_poses(
            self,
            posenet_model,
            output_stride: int,
            frame: List[List[float]],
            resolution: Tuple[int],
            dst_scores: float,
            dst_keypoints: float,
            model_type: str,
            score_threshold: float):
        """Predict relative coordinates, confident scores and validation masks
        for all detected poses

        Args:
            posenet model: tensorflow model
            output_stride (int): output stride to convert output indices to image
                coordinates
            frame (np.array): image for inference
            resolution (int): resolution to scale frame for inference
            dst_scores (np.array): 17x1 buffer to store keypoint scores
            dst_keypoints (np.array): 17x2 buffer to store keypoint coordinates
            model_type (str): specified model type (refer to modelconfig.yml)
            score_threshold (float): threshold for prediction

        Returns:
            full_keypoint_coords (np.array): keypoints coordinates of detected poses
            full_keypoint_scores (np.array): keypoints confidence scores of detected
                poses
            full_masks (np.array): keypoints validation masks of detected poses
        """
        input_res = resolution
        image, output_scale, image_size = self._create_image_from_frame(
            output_stride, frame, input_res, model_type)

        pose_count = detect_keypoints(
            posenet_model, image, output_stride, dst_scores, dst_keypoints,
            model_type, score_threshold)
        full_keypoint_scores = dst_scores[:pose_count]
        full_keypoint_coords = dst_keypoints[:pose_count]

        full_keypoint_rel_coords = get_keypoints_relative_coords(
            full_keypoint_coords, output_scale, image_size)

        full_masks = self._get_full_masks_from_keypoint_scores(full_keypoint_scores)

        return full_keypoint_rel_coords, full_keypoint_scores, full_masks

    @staticmethod
    def _create_image_from_frame(output_stride: int,
                                 frame: List[List[float]],
                                 input_res: int,
                                 model_type: str):
        """ Preprocess image for inference

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
    def _get_full_masks_from_keypoint_scores(keypoint_scores: List[List[float]]):
        """ PoseNet prediction function

        Args:
            keypoint_scores (np.array): keypoints confidence scores of detected poses

        Returns:
            masks (np.array): keypoints validation masks of detected poses
        """
        masks = keypoint_scores > MIN_PART_SCORE
        return masks
