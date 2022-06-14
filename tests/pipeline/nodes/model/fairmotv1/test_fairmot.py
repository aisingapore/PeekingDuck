from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml

from peekingduck.pipeline.nodes.base import WeightsDownloaderMixin
from peekingduck.pipeline.nodes.model.fairmot import Node
from peekingduck.pipeline.nodes.model.fairmotv1.fairmot_files.matching import (
    fuse_motion,
    iou_distance,
)
from tests.conftest import PKD_DIR

# Frame index for manual manipulation of detections to trigger some
# branches
SEQ_IDX = 6


@pytest.fixture
def fairmot_config():
    """Yields config while forcing the model to run on CPU."""
    with open(PKD_DIR / "configs" / "model" / "fairmot.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    with mock.patch("torch.cuda.is_available", return_value=False):
        yield node_config


@pytest.fixture
def fairmot_config_gpu():
    """Yields config which allows the model to run on GPU on CUDA-enabled
    devices.
    """
    with open(PKD_DIR / "configs" / "model" / "fairmot.yml") as infile:
        node_config = yaml.safe_load(infile)
    node_config["root"] = Path.cwd()

    yield node_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
        {"key": "K", "value": -0.5},
        {"key": "min_box_area", "value": -0.5},
        {"key": "track_buffer", "value": -0.5},
    ],
)
def fairmot_bad_config_value(request, fairmot_config):
    """Various invalid config values."""
    fairmot_config[request.param["key"]] = request.param["value"]
    return fairmot_config


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
class TestFairMOT:
    def test_should_give_empty_output_for_no_human_image(
        self, no_human_image, fairmot_config
    ):
        """Input images either contain nothing or non-humans."""
        no_human_img = cv2.imread(no_human_image)
        fairmot = Node(fairmot_config)
        output = fairmot.run({"img": no_human_img})
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
        self, human_video_sequence, fairmot_config
    ):
        _, detections = human_video_sequence
        fairmot = Node(fairmot_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            output = fairmot.run(inputs)
            if i > 1:
                for track_id, track in enumerate(fairmot.model.tracker.tracked_stracks):
                    assert repr(track) == f"OT_{track_id + 1}_(1-{i + 1})"
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_tracking_ids_should_be_consistent_across_frames_gpu(
        self, human_video_sequence, fairmot_config_gpu
    ):
        _, detections = human_video_sequence
        fairmot = Node(fairmot_config_gpu)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            output = fairmot.run(inputs)
            if i > 1:
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    def test_should_activate_unconfirmed_tracks_subsequently(
        self, human_video_sequence, fairmot_config
    ):
        _, detections = human_video_sequence
        fairmot = Node(fairmot_config)
        # Make frame_id start at 2 internally to avoid STrack from activating
        # in activate() when frame_id == 1
        fairmot.model.tracker.frame_id = 1
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            output = fairmot.run(inputs)
            if i > 1:
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    def test_reactivate_tracks(self, human_video_sequence, fairmot_config):
        _, detections = human_video_sequence
        fairmot = Node(fairmot_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            if i == SEQ_IDX:
                # These STrack should get re-activated
                for track in fairmot.model.tracker.tracked_stracks:
                    track.mark_lost()
            output = fairmot.run(inputs)
            if i > 1:
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    def test_associate_with_iou(self, human_video_sequence, fairmot_config):
        _, detections = human_video_sequence
        fairmot = Node(fairmot_config)
        prev_tags = []
        with mock.patch(
            "peekingduck.pipeline.nodes.model.fairmotv1.fairmot_files.matching.fuse_motion",
            wraps=replace_fuse_motion,
        ):
            for i, inputs in enumerate({"img": x["img"]} for x in detections):
                output = fairmot.run(inputs)
                if i > 1:
                    assert output["obj_attrs"]["ids"] == prev_tags
                prev_tags = output["obj_attrs"]["ids"]

    def test_mark_unconfirmed_tracks_for_removal(
        self, human_video_sequence, fairmot_config
    ):
        """Manipulate both embedding and iou distance to be above the cost
        limit so nothing gets associated and all gets marked for removal. As a
        result, the Tracker should not produce any track IDs.
        """
        _, detections = human_video_sequence
        fairmot = Node(fairmot_config)
        with mock.patch(
            "peekingduck.pipeline.nodes.model.fairmotv1.fairmot_files.matching.fuse_motion",
            wraps=replace_fuse_motion,
        ), mock.patch(
            "peekingduck.pipeline.nodes.model.fairmotv1.fairmot_files.matching.iou_distance",
            wraps=replace_iou_distance,
        ):
            for i, inputs in enumerate({"img": x["img"]} for x in detections):
                output = fairmot.run(inputs)
                if i == 0:
                    # Skipping the assert on the first frame. FairMOT sets
                    # STrack to is_activated=True on when frame_id=1 but JDE
                    # doesn't
                    continue
                assert not output["obj_attrs"]["ids"]

    def test_remove_lost_tracks(self, human_video_sequence, fairmot_config):
        # Set buffer and as a result `max_time_lost` to extremely short so
        # lost tracks will get removed
        _, detections = human_video_sequence
        fairmot_config["track_buffer"] = 1
        fairmot = Node(fairmot_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            if i >= SEQ_IDX:
                inputs["img"] = np.zeros_like(inputs["img"])
            output = fairmot.run(inputs)
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
    def test_new_video_frame_rate(
        self, human_video_sequence, fairmot_config, mot_metadata
    ):
        _, detections = human_video_sequence
        fairmot = Node(config=fairmot_config)
        prev_tags = []
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.fairmot_mot.fairmot_model.logger"
        ) as captured:
            for i, inputs in enumerate({"img": x["img"]} for x in detections):
                # Insert mot_metadata in input to signal a new model should be
                # created
                if i == 0:
                    inputs["mot_metadata"] = mot_metadata
                output = fairmot.run(inputs)
                if i == 0:
                    assert captured.records[0].getMessage() == (
                        "Creating new model with frame rate: "
                        f"{mot_metadata['frame_rate']:.2f}..."
                    )
                if i > 1:
                    assert output["obj_attrs"]["ids"] == prev_tags
                assert fairmot._frame_rate == pytest.approx(mot_metadata["frame_rate"])
                prev_tags = output["obj_attrs"]["ids"]

    def test_handle_empty_detections(
        self, human_video_sequence_with_empty_frames, fairmot_config
    ):
        _, detections = human_video_sequence_with_empty_frames
        fairmot = Node(fairmot_config)
        for i, inputs in enumerate(detections):
            output = fairmot.run(inputs)
            if i > 1:
                assert len(output["obj_attrs"]["ids"]) == len(inputs["bboxes"])

    def test_invalid_config_value(self, fairmot_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=fairmot_bad_config_value)
        assert "must be" in str(excinfo.value)

    @mock.patch.object(WeightsDownloaderMixin, "_has_weights", return_value=True)
    def test_invalid_config_model_files(self, _, fairmot_config):
        with pytest.raises(ValueError) as excinfo:
            fairmot_config["weights"][fairmot_config["model_format"]]["model_file"][
                "dla_34"
            ] = "some/invalid/path"
            _ = Node(config=fairmot_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, no_human_image, fairmot_config):
        no_human_img = cv2.imread(no_human_image)
        # Potentially passing in a file path or a tuple from image reader
        # output
        fairmot = Node(fairmot_config)
        with pytest.raises(TypeError) as excinfo:
            _ = fairmot.run({"img": Path.cwd()})
        assert str(excinfo.value) == "image must be a np.ndarray"
        with pytest.raises(TypeError) as excinfo:
            _ = fairmot.run({"img": ("image name", no_human_img)})
        assert str(excinfo.value) == "image must be a np.ndarray"
