# Modifications copyright 2022 AI Singapore
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
#
# Original copyright (c) 2019 ZhongdaoWang
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

"""JDE Multi-object Tracker.

Modifications include:
- Rearranged comments to they appear before the relevant lines of code
- Refactor variable names in update() for clarity
- Refactor subtract_stracks() to use list comprehension
- Refactor combine_stracks() to use bool for dictionary values instead
"""

import locale
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from peekingduck.nodes.model.jdev1.jde_files import matching
from peekingduck.nodes.model.jdev1.jde_files.darknet import Darknet
from peekingduck.nodes.model.jdev1.jde_files.kalman_filter import KalmanFilter
from peekingduck.nodes.model.jdev1.jde_files.track import STrack, TrackState
from peekingduck.nodes.model.jdev1.jde_files.utils import (
    letterbox,
    non_max_suppression,
    scale_coords,
)
from peekingduck.utils.bbox.transforms import tlwh2xyxyn, xyxy2tlwh


class Tracker:  # pylint: disable=too-many-instance-attributes
    """JDE Multi-object Tracker.

    Args:
        config (Dict[str, Any]): Model configuration options.
        model_dir (Path): Directory to model weights files.
        frame_rate (float): Frame rate of the current video sequence, used
            for computing size of track buffer.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        frame_rate: float,
        model_type: str,
        model_file: Dict[str, str],
        model_config_file: Dict[str, str],
        min_box_area: int,
        track_buffer: int,
        iou_threshold: float,
        nms_threshold: float,
        score_threshold: float,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_type = model_type
        self.model_path = model_dir / model_file[self.model_type]
        self.model_settings = self._parse_model_config(
            model_dir / model_config_file[self.model_type]
        )
        self.input_size = [
            int(self.model_settings[0]["width"]),
            int(self.model_settings[0]["height"]),
        ]
        self.min_box_area = min_box_area
        self.track_buffer = track_buffer
        self.iou_threshold = iou_threshold
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

        self.model = self._create_darknet_model()

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * self.track_buffer)

        self.kalman_filter = KalmanFilter()

    def track_objects_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[int], np.ndarray]:
        """Tracks detections from the current video frame.

        Args:
            image (np.ndarray): The current video frame.

        Returns:
            (Tuple[np.ndarray, List[int], np.ndarray]): A tuple of
            - Numpy array of detected bounding boxes.
            - List of track IDs.
            - List of detection confidence scores.
        """
        image_size = image.shape[:2]
        padded_image = self._preprocess(image)
        padded_image_tensor = (
            torch.from_numpy(padded_image).to(self.device).unsqueeze(0)
        )

        online_targets = self.update(padded_image_tensor, image)
        online_tlwhs = []
        online_ids = []
        scores = []
        for target in online_targets:
            tlwh = target.tlwh
            vertical = tlwh[2] / tlwh[3] > 1.6
            if not vertical and tlwh[2] * tlwh[3] > self.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(target.track_id)
                scores.append(target.score.item())
        if not online_tlwhs:
            return np.empty((0, 4)), online_ids, np.empty(0)

        bboxes = self._postprocess(np.array(online_tlwhs), image_size)

        return bboxes, online_ids, np.array(scores)

    @torch.no_grad()
    def update(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        self, padded_image: torch.Tensor, image: np.ndarray
    ) -> List[STrack]:
        """Processes the image frame and finds bounding box (detections).

        Associates the detection with corresponding tracklets and also handles
        lost, removed, re-found and active tracklets

        Args:
            padded_image (torch.Tensor): Preprocessed image with letterbox
                resizing and color normalization.
            image (np.ndarray): The original video frame.

        Returns:
            (List[STrack]): The list contains information regarding the
            online tracklets for the received image tensor.
        """
        self.frame_id += 1
        # For storing active tracks, for the current frame
        activated_stracks = []
        # Lost Tracks whose detections are obtained in the current frame
        refind_stracks = []
        # The tracks which are not obtained in the current frame but are not
        # removed. (Lost for some time less than the threshold for removing)
        lost_stracks = []
        removed_stracks = []

        # Step 1: Network forward, get detections & embeddings
        # pred is tensor of all the proposals (default number of proposals:
        # 54264). Proposals have information associated with the bounding box
        # and embeddings
        pred = self.model(padded_image)
        # Proposals rejected on basis of object confidence score
        pred = pred[pred[..., 4] > self.score_threshold]
        if len(pred) > 0:
            # Final proposals are obtained in dets. Information of bounding box
            # and embeddings also included
            dets = non_max_suppression(pred.unsqueeze(0), self.nms_threshold)[0].cpu()
            # Next step changes the detection scales
            scale_coords(
                self.input_size, dets[:, :4], (image.shape[0], image.shape[1])
            ).round()

            # Detections is list of (x1, y1, x2, y2, object_conf, class_score,
            # class_pred) class_pred is the embeddings.
            detections = [
                STrack(xyxy2tlwh(xyxys[:4]), xyxys[4], f.numpy())
                for (xyxys, f) in zip(dets[:, :5], dets[:, 6:])
            ]
        else:
            detections = []

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if track.is_activated:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)
            else:
                # previous tracks which are not active in the current frame
                # are added in unconfirmed list
                unconfirmed.append(track)

        # Step 2: First association, with embedding
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = _combine_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)

        # The dists is a matrix of distances of the detection with the tracks
        # in strack_pool
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # matches is the array for corresponding matches of the detection
        # with the corresponding strack_pool
        (
            matches,
            unmatched_track_indices,
            unmatched_det_indices,
        ) = matching.linear_assignment(dists, threshold=0.7)

        for tracked_idx, det_idx in matches:
            # tracked_idx is the id of the track and det_idx is the detection
            track = strack_pool[tracked_idx]
            det = detections[det_idx]
            if track.state == TrackState.TRACKED:
                # If the track is active, add the detection to the track
                track.update(detections[det_idx], self.frame_id)
                activated_stracks.append(track)
            else:
                # We have obtained a detection from a track which is not
                # active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        # Step 3: Second association, with IOU
        # detections is now a list of the unmatched detections
        detections = [detections[i] for i in unmatched_det_indices]
        # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        r_tracked_stracks = []
        for i in unmatched_track_indices:
            if strack_pool[i].state == TrackState.TRACKED:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        # matches is the list of detections which matched with corresponding
        # tracks by IOU distance method
        (
            matches,
            unmatched_track_indices,
            unmatched_det_indices,
        ) = matching.linear_assignment(dists, threshold=0.5)
        # Same process done for some unmatched detections, but now considering
        # IOU_distance as measure
        for tracked_idx, det_idx in matches:
            track = r_tracked_stracks[tracked_idx]
            det = detections[det_idx]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:  # pragma: no cover
                # This shouldn't be reached, r_tracked_stracks only takes in
                # tracks with TrackState.TRACKED from above
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)
        # If no detections are obtained for tracks (unmatched_track_indices),
        # the tracks are added to lost_tracks and are marked lost
        for i in unmatched_track_indices:
            track = r_tracked_stracks[i]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning
        # frame
        detections = [detections[i] for i in unmatched_det_indices]
        dists = matching.iou_distance(unconfirmed, detections)
        (
            matches,
            unconfirmed_track_indices,
            unmatched_det_indices,
        ) = matching.linear_assignment(dists, threshold=0.7)
        for tracked_idx, det_idx in matches:
            unconfirmed[tracked_idx].update(detections[det_idx], self.frame_id)
            activated_stracks.append(unconfirmed[tracked_idx])
        # The tracks which are yet not matched
        for i in unconfirmed_track_indices:
            track = unconfirmed[i]
            track.mark_removed()
            removed_stracks.append(track)
        # after all these confirmation steps, if a new detection is found, it
        # is initialized for a new track
        # Step 4: Init new stracks
        for i in unmatched_det_indices:
            track = detections[i]
            if track.score < self.score_threshold:  # pragma: no cover
                # This shouldn't be reached since we already rejected proposals
                # on basis of object confidence score earlier
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: Update state
        # If the tracks are lost for more frames than the threshold number, the
        # tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update the self.tracked_stracks and self.lost_stracks using the
        # updates in this step.
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.TRACKED
        ]
        self.tracked_stracks = _combine_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _combine_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = _subtract_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = _subtract_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = _remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

    def _create_darknet_model(self) -> Darknet:
        """Creates a Darknet-53 model corresponding specified `model_type`.

        Returns:
            (Darknet): Darknet backbone of the specified architecture and
                weights.
        """
        self.logger.info(
            "JDE model loaded with the following configs:\n\t"
            f"Model type: {self.model_type}\n\t"
            f"Input resolution: {self.input_size}\n\t"
            f"IOU threshold: {self.iou_threshold}\n\t"
            f"NMS threshold: {self.nms_threshold}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
            f"Min bounding box area: {self.min_box_area}\n\t"
            f"Track buffer: {self.track_buffer}"
        )
        return self._load_darknet_weights()

    def _load_darknet_weights(self) -> Darknet:
        """Loads pretrained Darknet-53 weights.

        Args:
            model_path (Path): Path to weights file.
            model_settings (List[Dict[str, Any]]): Model architecture
                configurations.

        Returns:
            (Darknet): Darknet backbone of the specified architecture and
                weights.
        """
        if not self.model_path.is_file():
            raise ValueError(
                f"Model file does not exist. Please check that {self.model_path} exists."
            )
        ckpt = torch.load(str(self.model_path), map_location="cpu")
        model = Darknet(self.model_settings, self.device, num_identities=14455)
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(self.device).eval()
        return model

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses the input image by padded resizing with letterbox and
        normalising RGB values.

        Args:
            image (np.ndarray): Input video frame.

        Returns:
            (np.ndarray): Preprocessed image.
        """
        # Padded resize
        padded_image = letterbox(
            image, height=self.input_size[1], width=self.input_size[0]
        )
        # Normalize RGB
        padded_image = padded_image[..., ::-1].transpose(2, 0, 1)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
        padded_image /= 255.0

        return padded_image

    @staticmethod
    def _parse_model_config(config_path: Path) -> List[Dict[str, Any]]:
        """Parse model configuration. Currently parses all values to string.

        Args:
            config_path (Path): Path to model configuration file.

        Returns:
            (List[Dict[str, Any]]): A list of dictionaries each containing
                the configuration of a layer/module.
        """
        with open(config_path, encoding=locale.getpreferredencoding(False)) as infile:
            lines = [
                line
                for line in map(str.strip, infile.readlines())
                if line and not line.startswith("#")
            ]
        module_defs: List[Dict[str, Any]] = []
        for line in lines:
            if line.startswith("["):
                module_defs.append({})
                module_defs[-1]["type"] = line[1:-1].rstrip()
                if module_defs[-1]["type"] == "convolutional":
                    module_defs[-1]["batch_normalize"] = 0
            else:
                key, value = tuple(map(str.strip, line.split("=")))
                if value.startswith("$"):
                    value = module_defs[0].get(value.strip("$"), None)
                module_defs[-1][key] = value
        return module_defs

    @staticmethod
    def _postprocess(tlwhs: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
        """Post-processes detection bounding boxes by converting them from
        [t, l, w, h] to normalized [x1, y1, x2, y2] format which is required by
        other PeekingDuck draw nodes. (t, l) is the top-left corner, w is
        width, and h is height. (x1, y1) is the top-left corner and (x2, y2) is
        the bottom-right corner.

        Args:
            tlwhs (np.ndarray): Bounding boxes in [t, l, w, h] format.
            image_shape (Tuple[int, ...]): Dimensions of the original video
                frame.

        Returns:
            (np.ndarray): Bounding boxes in normalized [x1, y1, x2, y2] format.
        """
        return tlwh2xyxyn(tlwhs, *image_shape)


def _combine_stracks(stracks_1: List[STrack], stracks_2: List[STrack]) -> List[STrack]:
    """Combines two list of STrack together.

    Args:
        stracks_1 (List[STrack]): List of STrack.
        stracks_2 (List[STrack]): List of STrack.

    Returns:
        (List[STrack]): Combined list of STrack.
    """
    stracks = {track.track_id: track for track in stracks_1}
    for track in stracks_2:
        tid = track.track_id
        if tid not in stracks:
            stracks[tid] = track
    return list(stracks.values())


def _remove_duplicate_stracks(
    stracks_1: List[STrack], stracks_2: List[STrack]
) -> Tuple[List[STrack], List[STrack]]:
    """Remove duplicate STrack based on costs computed using
    Intersection-over-Union (IoU) values. Duplicates are identified by
    cost<0.15, the STrack that is more recently created is marked as
    the duplicate.

    Args:
        stracks_1 (List[STrack]): List of STrack.
        stracks_2 (List[STrack]): List of STrack.

    Returns:
        (Tuple[List[STrack], List[STrack]]): Lists of STrack with duplicates
            removed.
    """
    distances = matching.iou_distance(stracks_1, stracks_2)
    pairs = np.where(distances < 0.15)
    duplicates_1 = []
    duplicates_2 = []
    for idx_1, idx_2 in zip(*pairs):
        age_1 = stracks_1[idx_1].frame_id - stracks_1[idx_1].start_frame
        age_2 = stracks_2[idx_2].frame_id - stracks_2[idx_2].start_frame
        if age_1 > age_2:
            duplicates_2.append(idx_2)
        else:
            duplicates_1.append(idx_1)
    return (
        [t for i, t in enumerate(stracks_1) if i not in duplicates_1],
        [t for i, t in enumerate(stracks_2) if i not in duplicates_2],
    )


def _subtract_stracks(stracks_1: List[STrack], stracks_2: List[STrack]) -> List[STrack]:
    """Removes stracks_2 from stracks_1.

    Args:
        stracks_1 (List[STrack]): List of STrack.
        stracks_2 (List[STrack]): List of STrack.

    Returns:
        (List[STrack]): List of STrack.
    """
    stracks = {track.track_id: track for track in stracks_1}
    for track in stracks_2:
        tid = track.track_id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())
