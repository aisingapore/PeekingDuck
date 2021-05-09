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
from typing import List
import numpy as np


class PoseData:
    """PoseData class containing information about predicted poses

       Args:
           keypoints (np.array): keypoints coordinates
           keypoint_scores (np.array): keypoints confidence scores
           masks (np.array): masks for low-confidence keypoints
           connections (np.array): list of connections
           activity (str): activity class name
    """

    def __init__(self,
                 keypoints: np.ndarray = None,
                 keypoint_scores: np.ndarray = None,
                 masks: np.ndarray = None,
                 connections: np.ndarray = None,
                 activity: str = ''):
        self.logger = logging.getLogger(__name__)
        self.bbox = None
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores
        self.masks = masks
        self.connections = connections
        self.activity = activity
        self.encode = None
