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

import logging

class PoseData:
    """A class contain Pose information predicted from a frame"""

    def __init__(self, keypoints=None, keypoint_scores=None, masks=None,
                 connections=None, activity=''):
        self.logger = logging.getLogger(__name__)

        self.keypoints = keypoints  # (np.array) (17 x 2) a list of keypoints' coordinates
        self.keypoint_scores = keypoint_scores  # (np.array) (17,) a list of scores
        self.masks = masks  # (np.array) (17) a list of masks for low-confidence keypoints
        self.connections = connections  # (float) (N x 2) a list of connections
        self.hand_bboxes = []  # (list of np.array) (2 x 2) bounding boxes for hand estimation
        self.hands = []  # a list of BusinessHand
        self.activity = activity  # (str) activity class name or other info like distance in OSL
        self.encode = None    # (float) a list of encode used in OSL visualisation only
