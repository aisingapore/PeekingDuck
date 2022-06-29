# Copyright 2022 AI Singapore
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

from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.jde import Node
from peekingduck.pipeline.nodes.model.jdev1.jde_files.matching import (
    fuse_motion,
    iou_distance,
)
from tests.conftest import PKD_DIR

# Frame index for manual manipulation of detections to trigger some
# branches
SEQ_IDX = 6


@pytest.fixture
def jde_config():
    """Yields config while forcing the model to run on CPU."""
    with open(PKD_DIR / "configs" / "model" / "jde.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    with mock.patch("torch.cuda.is_available", return_value=False):
        yield node_config


@pytest.fixture()
def jde_config_gpu():
    """Yields config which allows the model to run on GPU on CUDA-enabled
    devices.
    """
    with open(PKD_DIR / "configs" / "model" / "jde.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    yield node_config


@pytest.fixture(
    params=[
        {"key": "iou_threshold", "value": -0.5},
        {"key": "iou_threshold", "value": 1.5},
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
        {"key": "nms_threshold", "value": -0.5},
        {"key": "nms_threshold", "value": 1.5},
    ],
)
def jde_bad_config_value(request, jde_config):
    jde_config[request.param["key"]] = request.param["value"]
    return jde_config


def replace_fuse_motion(*args):
    """Manipulate the computed embedding distance so they are too large and
    cause none of the detections to be associated. This forces the Tracker to
    associate with IoU costs.
    """
    return np.ones_like(fuse_motion(*args))


def replace_iou_distance(*args):
    """Manipulate the computed IoU-based costs so they are too large and
    cause none of the detections to be associated. This forces the Tracker to
    mark tracks for removal.
    """
    return np.ones_like(iou_distance(*args))


@pytest.mark.mlmodel
class TestJDE:
    def test_no_human_image(self, no_human_image, jde_config):
        """Input images either contain nothing or non-humans."""
        no_human_img = cv2.imread(no_human_image)
        jde = Node(jde_config)
        output = jde.run({"img": no_human_img})
        expected_output = {
            "bboxes": [],
            "bbox_labels": [],
            "bbox_scores": [],
            "obj_attrs": {"ids": []},
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])
        npt.assert_equal(
            output["obj_attrs"]["ids"], expected_output["obj_attrs"]["ids"]
        )

    def test_tracking_ids_should_be_consistent_across_frames(
        self, human_video_sequence, jde_config
    ):
        """NOTE: This test includes testing the __repr__ of STrack which uses
        the **class** variable `track_id` (as opposed to instance variable). So
        this particular test has to be the first test to be run where
        detections get tracked, else it will fail.

        The class variable implemention of `track_id` follows the design of the
        original repo.
        """
        _, detections = human_video_sequence
        jde = Node(jde_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            output = jde.run(inputs)
            if i > 1:
                for track_id, track in enumerate(jde.model.tracker.tracked_stracks):
                    assert repr(track) == f"OT_{track_id + 1}_(1-{i + 1})"
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_tracking_ids_should_be_consistent_across_frames_gpu(
        self, human_video_sequence, jde_config_gpu
    ):
        _, detections = human_video_sequence
        jde = Node(jde_config_gpu)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            output = jde.run(inputs)
            if i > 1:
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    def test_detect_human_bboxes(self, human_video, jde_config):
        jde = Node(jde_config)
        frame_id = 0
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(human_video)
        ret, frame = cap.read()
        while ret:
            output = jde.run({"img": frame})
            if frame_id > 0:
                assert len(output["bboxes"]) == 3
                assert len(output["bbox_labels"]) == 3
            ret, frame = cap.read()
            frame_id += 1
        # When everything done, release the video capture object
        cap.release()

    def test_reactivate_tracks(self, human_video_sequence, jde_config):
        _, detections = human_video_sequence
        jde = Node(jde_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            if i == SEQ_IDX:
                # These STrack should get re-activated
                for track in jde.model.tracker.tracked_stracks:
                    track.mark_lost()
            output = jde.run(inputs)
            if i > 1:
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    def test_associate_with_iou(self, human_video_sequence, jde_config):
        _, detections = human_video_sequence
        jde = Node(jde_config)
        prev_tags = []
        with mock.patch(
            "peekingduck.pipeline.nodes.model.jdev1.jde_files.matching.fuse_motion",
            wraps=replace_fuse_motion,
        ):
            for i, inputs in enumerate({"img": x["img"]} for x in detections):
                output = jde.run(inputs)
                if i > 1:
                    assert output["obj_attrs"]["ids"] == prev_tags
                prev_tags = output["obj_attrs"]["ids"]

    def test_mark_unconfirmed_tracks_for_removal(
        self, human_video_sequence, jde_config
    ):
        """Manipulate both embedding and iou distance to be above the cost
        limit so nothing gets associated and all gets marked for removal. As a
        result, the Tracker should no produce any track IDs.
        """
        _, detections = human_video_sequence
        jde = Node(jde_config)
        with mock.patch(
            "peekingduck.pipeline.nodes.model.jdev1.jde_files.matching.fuse_motion",
            wraps=replace_fuse_motion,
        ), mock.patch(
            "peekingduck.pipeline.nodes.model.jdev1.jde_files.matching.iou_distance",
            wraps=replace_iou_distance,
        ):
            for inputs in ({"img": x["img"]} for x in detections):
                output = jde.run(inputs)
                assert not output["obj_attrs"]["ids"]

    def test_remove_lost_tracks(self, human_video_sequence, jde_config):
        _, detections = human_video_sequence
        # Set buffer and as a result `max_time_lost` to extremely short so
        # lost tracks will get removed
        jde_config["track_buffer"] = 1
        jde = Node(jde_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            if i >= SEQ_IDX:
                inputs["img"] = np.zeros_like(inputs["img"])
            output = jde.run(inputs)
            # switched to black image from SEQ_IDX onwards, nothing should be
            # detected on this frame ID
            if i == SEQ_IDX:
                assert not output["obj_attrs"]["ids"]
            elif i > 1:
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    @pytest.mark.parametrize(
        "mot_metadata",
        [
            {"frame_rate": 30.0, "reset_model": True},
            {"frame_rate": 10.0, "reset_model": False},
        ],
    )
    def test_reset_model(self, human_video_sequence, jde_config, mot_metadata):
        """_reset_model() should be called when either the frame_rate changes
        or when reset_model is True.
        """
        _, detections = human_video_sequence
        jde = Node(config=jde_config)
        prev_tags = []
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.jde.logger"
        ) as captured:
            for i, inputs in enumerate({"img": x["img"]} for x in detections):
                # Insert mot_metadata in input to signal a new model should be
                # created
                if i == 0:
                    inputs["mot_metadata"] = mot_metadata
                output = jde.run(inputs)
                if i == 0:
                    assert captured.records[0].getMessage() == (
                        "Creating new model with frame rate: "
                        f"{mot_metadata['frame_rate']:.2f}..."
                    )
                if i > 1:
                    assert output["obj_attrs"]["ids"] == prev_tags
                assert jde._frame_rate == pytest.approx(mot_metadata["frame_rate"])
                prev_tags = output["obj_attrs"]["ids"]

    def test_handle_empty_detections(
        self, human_video_sequence_with_empty_frames, jde_config
    ):
        _, detections = human_video_sequence_with_empty_frames
        jde = Node(jde_config)
        for i, inputs in enumerate(detections):
            output = jde.run(inputs)
            if i > 1:
                assert len(output["obj_attrs"]["ids"]) == len(inputs["bboxes"])

    def test_invalid_config_value(self, jde_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=jde_bad_config_value)
        assert "_threshold must be between [0.0, 1.0]" in str(excinfo.value)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, jde_config):
        with pytest.raises(ValueError) as excinfo:
            jde_config["weights"][jde_config["model_format"]]["model_file"][
                jde_config["model_type"]
            ] = "some/invalid/path"
            _ = Node(config=jde_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, no_human_image, jde_config):
        no_human_img = cv2.imread(no_human_image)
        # Potentially passing in a file path or a tuple from image reader
        # output
        jde = Node(jde_config)
        with pytest.raises(TypeError) as excinfo:
            _ = jde.run({"img": Path.cwd()})
        assert str(excinfo.value) == "image must be a np.ndarray"
        with pytest.raises(TypeError) as excinfo:
            _ = jde.run({"img": ("image name", no_human_img)})
        assert str(excinfo.value) == "image must be a np.ndarray"
