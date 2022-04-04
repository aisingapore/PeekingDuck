# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import yaml

from peekingduck.pipeline.nodes.base import (
    PEEKINGDUCK_WEIGHTS_SUBDIR,
    WeightsDownloaderMixin,
)
from peekingduck.pipeline.nodes.model.hrnet import Node
from tests.conftest import PKD_DIR, do_nothing


@pytest.fixture
def hrnet_config():
    with open(PKD_DIR / "configs" / "model" / "hrnet.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    return node_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def hrnet_bad_config_value(request, hrnet_config):
    hrnet_config[request.param["key"]] = request.param["value"]
    return hrnet_config


@pytest.mark.mlmodel
class TestHrnet:
    def test_no_human_image(self, test_no_human_images, hrnet_config):
        """Tests HRnet on images with no humans present."""
        blank_image = cv2.imread(test_no_human_images)
        hrnet = Node(hrnet_config)
        output = hrnet.run({"img": blank_image, "bboxes": np.empty((0, 4))})
        expected_output = {
            "keypoints": np.zeros(0),
            "keypoint_scores": np.zeros(0),
            "keypoint_conns": np.zeros(0),
        }

        assert output.keys() == expected_output.keys(), "missing keys"
        for i in expected_output.keys():
            npt.assert_array_equal(
                output[i], expected_output[i], err_msg=f"unexpected output for {i}"
            )

    def test_return_at_least_one_person_and_one_bbox(
        self, test_human_images, hrnet_config
    ):
        """Tests HRnet on images with at least one human present. Bbox
        coordinates is set as the entire image.
        """
        test_img = cv2.imread(test_human_images)
        img_h, img_w, _ = test_img.shape
        hrnet = Node(hrnet_config)
        output = hrnet.run(
            {"img": test_img, "bboxes": np.array([[0, 0, img_w, img_h]])}
        )

        assert "keypoints" in output
        assert "keypoint_scores" in output
        assert "keypoint_conns" in output
        assert output["keypoints"].size != 0

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=False)
    @mock.patch.object(WeightsDownloaderMixin, "_download_blob_to", wraps=do_nothing)
    @mock.patch.object(WeightsDownloaderMixin, "extract_file", wraps=do_nothing)
    def test_no_weights(
        self,
        _,
        mock_download_blob_to,
        mock_extract_file,
        hrnet_config,
    ):
        weights_dir = hrnet_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.hrnetv1.hrnet_model.logger"
        ) as captured:
            hrnet = Node(config=hrnet_config)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert hrnet is not None

        assert mock_download_blob_to.called
        assert mock_extract_file.called

    def test_invalid_config_value(self, hrnet_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=hrnet_bad_config_value)
        assert "_threshold must be between [0, 1]" in str(excinfo.value)
