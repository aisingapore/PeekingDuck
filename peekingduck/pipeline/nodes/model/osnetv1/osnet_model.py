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
#
# Original copyright (c) 2018 Kaiyang Zhou
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
OSNet Inference.
"""

from typing import Any, Dict
from pathlib import Path
import os
import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from peekingduck.weights_utils import checker, downloader, finder
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.crop import crop_bbox
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.distance import (
    compute_distance_matrix,
)
from peekingduck.pipeline.nodes.model.osnetv1.osnet_files.feature_extractor import (
    FeatureExtractor,
)


class OSNetModel:  # pylint: disable=too-few-public-methods
    """OSNet re-ID model for person identification."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Check threshold values
        if not 0 <= config["multi_threshold"] <= 1:
            raise ValueError("multi_threshold must be in [0, 1]")

        query_root_dir = config["query_root_dir"]
        self.multi_threshold = config["multi_threshold"]
        model_names = {
            "osnet": "osnet_x1_0",
            "osnet_ain": "osnet_ain_x1_0",
        }

        # Get names of folders in root query directory
        persons = os.listdir(query_root_dir)
        if not persons:
            raise ValueError(
                "There must be at least 1 folder in query_root_dir with \
                images to be queried."
            )

        weights_dir, model_dir = finder.find_paths(
            config["root"], config["weights"], config["weights_parent_dir"]
        )

        # check for OSNet weights, if none then download
        if not checker.has_weights(weights_dir, model_dir):
            self.logger.info("No weights detected. Proceeding to download...")
            downloader.download_weights(weights_dir, config["weights"]["blob_file"])
            self.logger.info(f"Weights downloaded to {weights_dir}.")

        model_path = model_dir / config["weights"]["model_file"][config["model_type"]]
        # Initialize model for extracting features
        self.extractor = FeatureExtractor(
            model_name=model_names[config["model_type"]],
            model_path=str(model_path),
            device=config["device"],
        )
        self.logger.info(
            f"Model name: {model_names[config['model_type']]}\n"
            " - params: 2,193,616\n"
            " - flops: 978,878,352"
        )

        # Create dict to store tag (folder) name and queried features
        self.queries = {}
        for folder in persons:
            # Feature extractor takes list of image paths
            query_img_paths = []
            path = query_root_dir + "/" + folder
            path = Path(path)
            # Get file paths in root/person1/*
            for file_path in path.glob("*"):
                query_img_paths.append(str(file_path))
            # Extract features from query image list
            self.queries[folder] = self.extractor(query_img_paths)
            self.logger.info(f"Features extracted for images in {folder} folder")

    # pylint: disable=too-many-locals
    def predict(self, img: np.ndarray, bboxes: np.ndarray) -> Dict[str, np.ndarray]:
        """This node returns the matching bbox of the queried person
        from the bboxes in the frame.

        Args:
            img (np.array): Image from video frame.
            bboxes (np.array): Detected bboxes of persons.

        Returns:
            match_info (dict): Bounding box of detected person(s).
        """
        frame = np.copy(img)
        original_h, original_w, _ = frame.shape
        bboxes = np.copy(bboxes)

        # Crop detected persons from frame
        cropped_bboxes = []
        for bbox in bboxes:
            cropped_bbox = crop_bbox(frame, bbox, original_h, original_w)
            cropped_bboxes.append(cropped_bbox)

        # cropped_bboxes list cannot be empty
        if cropped_bboxes:
            # Extract features from cropped images (gallery)
            gallery_features = self.extractor(cropped_bboxes)
            cost_distance = {}
            for name, qfeature in self.queries.items():
                # Calculate distance matching between query and gallery features
                distmat = compute_distance_matrix(
                    qfeature, gallery_features, metric="cosine"
                )
                distmat = distmat.cpu().numpy()
                if len(self.queries) == 1:
                    match_info = self._single_person(distmat, bboxes, name)
                else:  # Case when queries are > 1
                    # Create cost matrix
                    cost_distance[name] = distmat
            if len(self.queries) > 1:
                match_info = self._multi_person(cost_distance, bboxes)
        else:
            match_info = {}

        return match_info

    @staticmethod
    def _single_person(
        distmat: np.ndarray, bboxes: np.ndarray, name: str
    ) -> Dict[str, np.ndarray]:
        """Performs matching for a single queried person based on cosine
        distance and indices.

        Args:
            distmat (np.ndarray): Distance matrix.
            bboxes (np.ndarray): Detected bounding boxes of persons in frame.
            name (str): Folder name of queried person.

        Returns:
            Dict[str, np.ndarray]: Bounding box of detected person.
        """
        match_info = {}
        indices = np.argsort(distmat, axis=1)
        index = np.argmin(indices)  # cosine distance = (1 - similarity)
        match_bbox = bboxes[index]
        match_info[name] = match_bbox

        return match_info

    def _multi_person(
        self, cost_distance: Dict[str, Any], bboxes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Performs matching for multiple queried persons using the
        Hungarian Algorithm.

        Args:
            cost_distance (Dict[str, Any]): Dict containing folder name
                of queried person and their respective distance matrices.
            bboxes (np.ndarray): Detected bounding boxes of persons in frames.

        Returns:
            Dict[str, np.ndarray]: Bounding box of detected persons.
        """
        match_info = {}
        names = list(cost_distance.keys())
        cost_matrix_list = list(cost_distance.values())
        # Convert list of arrays to single array
        cost_matrix = np.concatenate(cost_matrix_list)
        # Perform Hungarian Algorithm matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for name, row_idx, col_idx in zip(names, row_ind, col_ind):
            cost = cost_matrix[row_idx, col_idx]
            # Comparing cosine distance against threshold
            if cost <= self.multi_threshold:
                match_bbox = bboxes[col_idx]
                match_info[name] = match_bbox
            else:
                match_info = {}

        return match_info
