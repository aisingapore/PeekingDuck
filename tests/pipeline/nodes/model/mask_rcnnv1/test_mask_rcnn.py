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
from unittest import mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.mask_rcnn import Node
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn
from tests.conftest import PKD_DIR, get_groundtruth

GT_RESULTS = get_groundtruth(Path(__file__).resolve())
NP_FILE = np.load(Path(__file__).resolve().parent / "mask_rcnn_gt_masks.npz")


@pytest.fixture
def mask_rcnn_config():
    with open(PKD_DIR / "configs" / "model" / "mask_rcnn.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = PKD_DIR
    node_config["iou_threshold"] = 0.5
    node_config["score_threshold"] = 0.5
    node_config["mask_threshold"] = 0.5

    # test on CPU only
    with mock.patch("torch.cuda.is_available", return_value=False):
        yield node_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
        {"key": "mask_threshold", "value": -0.5},
        {"key": "mask_threshold", "value": 1.5},
        {"key": "iou_threshold", "value": -0.5},
        {"key": "iou_threshold", "value": 1.5},
        {"key": "min_size", "value": 0},
        {"key": "max_size", "value": 0},
        {"key": "max_num_detections", "value": 0},
    ],
)
def mask_rcnn_bad_config_value(request, mask_rcnn_config):
    mask_rcnn_config[request.param["key"]] = request.param["value"]
    return mask_rcnn_config


@pytest.fixture(params=["r50-fpn", "r101-fpn"])
def mask_rcnn_type(request, mask_rcnn_config):
    mask_rcnn_config["model_type"] = request.param
    return mask_rcnn_config


@pytest.mark.mlmodel
class TestMaskRCNN:
    def test_no_human_image(self, no_human_image, mask_rcnn_type):
        no_human_img = cv2.imread(no_human_image)
        mask_rcnn_type["score_threshold"] = 0.9
        mask_rcnn = Node(config=mask_rcnn_type)
        output = mask_rcnn.run({"img": no_human_img})
        expected_output = {
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_labels": np.empty((0)),
            "bbox_scores": np.empty((0), dtype=np.float32),
            "masks": np.empty((0, 0, 0), dtype=np.uint8),
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])
        npt.assert_equal(output["masks"], expected_output["masks"])

    def test_detect_human_bboxes(self, human_image, mask_rcnn_type):
        human_img = cv2.imread(human_image)
        image_size = human_img.size
        mask_rcnn = Node(config=mask_rcnn_type)
        output = mask_rcnn.run({"img": human_img})

        assert "bboxes" in output
        assert output["bboxes"].size > 0

        assert "masks" in output
        assert output["masks"].size > 0

        model_type = mask_rcnn.config["model_type"]
        image_name = Path(human_image).stem
        expected = GT_RESULTS[model_type][image_name]
        expected_mask = NP_FILE[f"{model_type}_{image_name}"]
        npt.assert_allclose(output["bboxes"], expected["bboxes"], atol=1e-3)
        npt.assert_equal(output["bbox_labels"], expected["bbox_labels"])
        npt.assert_allclose(output["bbox_scores"], expected["bbox_scores"], atol=1e-2)

        # maximum percentage of allowable difference in mask pixel values
        perc_pixel_diff_tol = 5e-4
        assert (
            np.sum(output["masks"] != expected_mask) / image_size <= perc_pixel_diff_tol
        )

    def test_mask_rcnn_preprocess(self, create_image, mask_rcnn_config):
        test_img = create_image((720, 1280, 3))
        mask_rcnn = Node(config=mask_rcnn_config)

        actual_img = mask_rcnn.model.detector._preprocess(test_img)

        assert isinstance(actual_img, (list, torch.Tensor))
        if isinstance(actual_img, list):
            assert isinstance(actual_img[0], torch.Tensor)
            assert len(actual_img[0].shape) == 3
        else:
            assert len(actual_img.shape) == 4

        assert actual_img[0].shape[0] == 3
        assert actual_img[0].dtype == torch.float32

    def test_mask_rcnn_postprocess(self, mask_rcnn_config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_shape = (3, 3)

        # boxes: [x1, y1, x2, y2] where 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        # 3.2 in [0.0, 1.0, 1.0, 3.2] is to test for clipping
        network_output = {
            "boxes": torch.tensor(
                [[0.0, 1.0, 1.0, 3.2], [1.0, 0.0, 2.0, 2.0]],
                dtype=torch.float32,
                device=device,
            ),
            "labels": torch.tensor([1, 35], dtype=torch.int64, device=device),
            "scores": torch.tensor([0.9, 0.2], dtype=torch.float32, device=device),
            "masks": torch.rand(
                (2, 1, img_shape[0], img_shape[1]),
                dtype=torch.float32,
                device=device,
            ),
        }

        expected_masks = network_output["masks"][0].clone()

        mask_rcnn = Node(config=mask_rcnn_config)
        boxes, labels, scores, masks = mask_rcnn.model.detector._postprocess(
            network_output, img_shape
        )
        # All outputs should be numpy arrays
        for item in (boxes, labels, scores, masks):
            assert isinstance(item, np.ndarray)
        assert masks.dtype == np.uint8

        expected_bbox = xyxy2xyxyn(
            np.array([[0.0, 1.0, 1.0, 3.0]], dtype=np.float32),
            height=img_shape[0],
            width=img_shape[1],
        )

        expected_score = np.array([0.9])

        expected_masks = expected_masks > mask_rcnn_config["mask_threshold"]
        expected_masks = expected_masks.squeeze(1).cpu().numpy().astype(np.uint8)

        npt.assert_almost_equal(expected_bbox, boxes)
        npt.assert_almost_equal(expected_score, scores)
        npt.assert_equal(expected_masks, masks)
        # Only one expected instance (person) because default detect should only
        # be person only ``[0]``
        npt.assert_equal(np.array(["person"]), labels)

    def test_invalid_config_value(self, mask_rcnn_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=mask_rcnn_bad_config_value)
        assert "must be" in str(excinfo.value)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, mask_rcnn_config):
        with pytest.raises(FileNotFoundError) as excinfo:
            mask_rcnn_config["weights"][mask_rcnn_config["model_format"]]["model_file"][
                mask_rcnn_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=mask_rcnn_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, no_human_image, mask_rcnn_config):
        no_human_img = cv2.imread(no_human_image)
        mask_rcnn = Node(config=mask_rcnn_config)
        # Potentially passing in a file path or a tuple from image reader
        # output
        with pytest.raises(TypeError) as excinfo:
            _ = mask_rcnn.run({"img": Path.cwd()})
        assert "image must be a np.ndarray" == str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            _ = mask_rcnn.run({"img": ("image name", no_human_img)})
        assert "image must be a np.ndarray" == str(excinfo.value)

    def test_invalid_detect_id(self, mask_rcnn_config):
        mask_rcnn_config["detect"] = 0
        with pytest.raises(TypeError) as excinfo:
            # Passing a non-list detect_id into the config
            _ = Node(config=mask_rcnn_config)
        assert "detect_ids has to be a list" == str(excinfo.value)
