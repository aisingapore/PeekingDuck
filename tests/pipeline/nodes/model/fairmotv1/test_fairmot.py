from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml

from peekingduck.pipeline.nodes.model.fairmot import Node
from peekingduck.pipeline.nodes.model.fairmotv1.fairmot_files.matching import (
    fuse_motion,
    iou_distance,
)
from peekingduck.weights_utils.finder import PEEKINGDUCK_WEIGHTS_SUBDIR

# Frame index for manual manipulation of detections to trigger some
# branches
SEQ_IDX = 6


@pytest.fixture
def fairmot_config():
    """Yields config while forcing the model to run on CPU."""
    file_path = Path(__file__).resolve().parent / "test_fairmot.yml"
    with open(file_path) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    with mock.patch("torch.cuda.is_available", return_value=False):
        yield node_config


@pytest.fixture
def fairmot_config_gpu():
    """Yields config which allows the model to run on GPU on CUDA-enabled
    devices.
    """
    file_path = Path(__file__).resolve().parent / "test_fairmot.yml"
    with open(file_path) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()

    yield node_config


@pytest.fixture(
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def fairmot_bad_config_value(request, fairmot_config):
    """Various invalid config values."""
    fairmot_config[request.param["key"]] = request.param["value"]
    return fairmot_config


@pytest.fixture(
    params=[
        {"key": "K", "value": -0.5},
        {"key": "min_box_area", "value": -0.5},
        {"key": "track_buffer", "value": -0.5},
    ],
)
def fairmot_negative_config_value(request, fairmot_config):
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
    def test_should_give_empty_output_for_no_human_images(
        self, test_no_human_images, fairmot_config
    ):
        """Input images either contain nothing or non-humans."""
        blank_image = cv2.imread(test_no_human_images)
        fairmot = Node(fairmot_config)
        output = fairmot.run({"img": blank_image})
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
        self, test_human_video_sequences, fairmot_config
    ):
        _, detections = test_human_video_sequences
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
        self, test_human_video_sequences, fairmot_config_gpu
    ):
        _, detections = test_human_video_sequences
        fairmot = Node(fairmot_config_gpu)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in detections):
            output = fairmot.run(inputs)
            if i > 1:
                assert output["obj_attrs"]["ids"] == prev_tags
            prev_tags = output["obj_attrs"]["ids"]

    def test_should_activate_unconfirmed_tracks_subsequently(
        self, test_human_video_sequences, fairmot_config
    ):
        _, detections = test_human_video_sequences
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

    def test_reactivate_tracks(self, test_human_video_sequences, fairmot_config):
        _, detections = test_human_video_sequences
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

    def test_associate_with_iou(self, test_human_video_sequences, fairmot_config):
        _, detections = test_human_video_sequences
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
        self, test_human_video_sequences, fairmot_config
    ):
        """Manipulate both embedding and iou distance to be above the cost
        limit so nothing gets associated and all gets marked for removal. As a
        result, the Tracker should not produce any track IDs.
        """
        _, detections = test_human_video_sequences
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

    def test_remove_lost_tracks(self, test_human_video_sequences, fairmot_config):
        # Set buffer and as a result `max_time_lost` to extremely short so
        # lost tracks will get removed
        _, detections = test_human_video_sequences
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
        self, test_human_video_sequences, fairmot_config, mot_metadata
    ):
        _, detections = test_human_video_sequences
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

    def test_invalid_config_value(self, fairmot_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=fairmot_bad_config_value)
        assert "_threshold must be in [0, 1]" in str(excinfo.value)

    def test_negative_config_value(self, fairmot_negative_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=fairmot_negative_config_value)
        assert "must be more than 0" in str(excinfo.value)

    def test_invalid_config_model_files(self, fairmot_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=True
        ), pytest.raises(ValueError) as excinfo:
            fairmot_config["weights"]["model_file"]["dla_34"] = "some/invalid/path"
            _ = Node(config=fairmot_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, test_no_human_images, fairmot_config):
        blank_image = cv2.imread(test_no_human_images)
        # Potentially passing in a file path or a tuple from image reader
        # output
        fairmot = Node(fairmot_config)
        with pytest.raises(TypeError) as excinfo:
            _ = fairmot.run({"img": Path.cwd()})
        assert str(excinfo.value) == "image must be a np.ndarray"
        with pytest.raises(TypeError) as excinfo:
            _ = fairmot.run({"img": ("image name", blank_image)})
        assert str(excinfo.value) == "image must be a np.ndarray"

    def test_no_weights(self, fairmot_config, replace_download_weights):
        weights_dir = fairmot_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.fairmot_mot.fairmot_model.logger"
        ) as captured:
            fairmot = Node(config=fairmot_config)
            print(captured)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert fairmot is not None
