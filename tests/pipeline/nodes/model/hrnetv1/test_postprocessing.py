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

import numpy as np
import numpy.testing as npt

from peekingduck.pipeline.nodes.model.hrnetv1.hrnet_files.postprocessing import (
    get_valid_keypoints,
    reshape_heatmaps,
    scale_transform,
)

SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]


class TestPostprocessing:
    def test_scale_transform(self):
        test_arr = np.array([[36, 6], [60, 40], [5, 20], [30, 45]])
        test_in_scale = (64, 48)
        test_out_scale = (720, 480)

        expected_output = test_arr * (
            np.array(test_out_scale) / np.array(test_in_scale)
        )
        actual_output = scale_transform(test_arr, test_in_scale, test_out_scale)

        npt.assert_almost_equal(expected_output, actual_output)

    def test_reshape_heatmaps(self):
        test_heatmap = np.random.rand(5, 48, 64, 17)

        output = reshape_heatmaps(test_heatmap)
        assert output.shape == (5, 17, 3072)

    def test_get_valid_keypoints(self):
        test_arr = np.random.rand(2, 17, 2)
        test_kp_scores = np.vstack((np.ones((1, 17)), np.zeros((1, 17))))
        test_batch, _, _ = test_arr.shape
        test_min_score = 0.2
        output_kp, output_masks = get_valid_keypoints(
            test_arr, test_kp_scores, test_batch, test_min_score
        )

        expected_out_arr = np.vstack((test_arr[0, :, :], np.zeros((17, 2)))).reshape(
            test_batch, 17, -1
        )
        expected_out_masks = test_kp_scores > test_min_score

        assert expected_out_arr.shape == output_kp.shape
        npt.assert_almost_equal(expected_out_arr, output_kp)
        npt.assert_almost_equal(expected_out_masks, output_masks)
