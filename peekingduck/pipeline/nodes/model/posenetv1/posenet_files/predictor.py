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
import tensorflow as tf
import logging
import numpy as np

from .posedata import PoseData
from .detector import detect_keypoints_yx, \
    get_keypoints_relative_coords, SCALE_FACTOR, KEYPOINTS_NUM
from .preprocessing import rescale_image
from .graph_functions import load_graph

class Predictor:

    def __init__(self, config):

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config['root']
        self.output_stride = self.config['output_stride']
        self.resolution = self.get_resolution_as_tuple(self.config['resolution'])
        self.max_pose_detection = self.config['max_pose_detection']
        self.model_type = self.config['model_type']
        self.score_threshold = self.config['score_threshold']

        self.logger.info('load model [posenet#%s]', self.model_type)

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

    def get_resolution_as_tuple(self, resolution):
        '''
        Takes a resolution in str format seperated by 'x'and convert into
        tuple. e.g. 225x225 returns (225, 225)
        '''

        res1, res2 = resolution.split('x')

        return (int(res1), int(res2))


    def predict(self, frame):
        """Predict pose and update the business frame
        Note: full means all poses

        args:
            posenet_model - the posenet mode's function
            output_stride - by default it is 16
            frame - frame to perform inference with
            resolution - resolution to scale frame to for inference
            dst_scores - 17x1 buffer to store keypoint scores
            dst_keypoints - 17 x 2 buffer to store keypoint coordinates
            model_type - model type specified in modelconfig.yml
            score_threshold - the threshold for predictions

        return:
            business_frame
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

        for coords, scores, masks in zip(full_keypoint_rel_coords, full_keypoint_scores, full_masks):
            pose = PoseData(keypoints=coords, keypoint_scores=scores,
                            masks=masks, connections=None)
            poses.append(pose)

        return poses


    def _predict_all_poses(
            self,
            posenet_model,
            output_stride,
            frame,
            resolution,
            dst_scores,
            dst_keypoints,
            model_type,
            score_threshold):
        """Predict relative coordinates, confident scores and validation masks for
        all keypoints for all poses detected from image.  We call 'full' because
        they are for all poses

        args:
            posenet_model - tensorflow model
            output_stride - output stride to convert output indices to image coords
            frame - frame to perform inference with
            resolution - resolution to scale frame to for inference
            dst_scores - 17x1 buffer to store keypoint scores
            dst_keypoints - 17 x 2 buffer to store keypoint coordinates
            model_type - model type specified in modelconfig.yml
            score_threshold - the threshold for predictions

        return:
            - full_keypoint_coords: all poses' keypoints' coordinates
            - full_keypoint_scores: all poses' keypoints' confidence scores
            - full_masks: all poses' keypoints' validation masks
        """
        input_res = resolution
        image, output_scale, image_size = self._create_image_from_frame(
            output_stride, frame, input_res, model_type)

        pose_count = detect_keypoints_yx(
            posenet_model, image, output_stride, dst_scores, dst_keypoints,
            model_type, score_threshold)
        full_keypoint_scores = dst_scores[:pose_count]
        full_keypoint_coords = dst_keypoints[:pose_count]

        full_keypoint_rel_coords = get_keypoints_relative_coords(
            full_keypoint_coords, output_scale, image_size)

        full_masks = self._get_full_masks_from_keypoint_scores(full_keypoint_scores)

        return full_keypoint_rel_coords, full_keypoint_scores, full_masks


    def _create_image_from_frame(self, output_stride, frame, input_res, model_type):
        image_size = [frame.shape[1], frame.shape[0]]

        image, output_scale = rescale_image(frame,
                                            input_res,
                                            scale_factor=SCALE_FACTOR,
                                            output_stride=output_stride,
                                            model_type=model_type
                                            )
        image = tf.convert_to_tensor(image)
        return image, output_scale, image_size


    def _get_full_masks_from_keypoint_scores(self, keypoint_scores):
        # https://github.com/rwightman/posenet-python/blob/master/webcam_demo.py#L55
        MIN_PART_SCORE = 0.1
        masks = keypoint_scores > MIN_PART_SCORE
        return masks
