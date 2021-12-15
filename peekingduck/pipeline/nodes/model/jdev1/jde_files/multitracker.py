# Modifications copyright 2021 AI Singapore
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

"""
Script for creating Track and JDE model instance.
"""

from collections import deque
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from peekingduck.pipeline.nodes.model.jdev1.jde_files.models import (
    Darknet,
)
from peekingduck.pipeline.nodes.model.jdev1.jde_files import matching
from peekingduck.pipeline.nodes.model.jdev1.jde_files.basetrack import (
    BaseTrack,
    TrackState,
)
from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils.kalman_filter import (
    KalmanFilter,
)
from peekingduck.pipeline.nodes.model.jdev1.jde_files.utils.utils import (
    non_max_suppression,
    scale_coords,
)


# pylint: disable=too-many-instance-attributes
class STrack(BaseTrack):
    """STrack class."""

    def __init__(
        self,
        tlwh: torch.Tensor,
        score: torch.Tensor,
        temp_feat: torch.Tensor,
        buffer_size: int = 30,
    ) -> None:
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score  # type: ignore
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)  # type: ignore
        self.alpha = 0.9

    def update_features(self, feat: np.ndarray) -> None:
        """Update features.

        Args:
            feat (np.ndarray): Features of predictions.
        """
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self) -> None:
        """Predict function."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks: List[Any], kalman_filter: KalmanFilter) -> None:
        """Multi predict for STracks.

        Args:
            stracks (List[Any]): List of STracks.
            kalman_filter (KalmanFilter): Kalman filter for state estimation.
        """
        if stracks:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, track in enumerate(stracks):
                if track.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = kalman_filter.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    # pylint: disable=arguments-differ
    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Start a new tracklet.

        Args:
            kalman_filter (KalmanFilter): Kalman filter for state estimation.
            frame_id (int): ID for current frame.
        """
        self.kalman_filter = kalman_filter  # type: ignore
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(
        self, new_track: List[Any], frame_id: int, new_id: bool = False
    ) -> None:
        """Re-activate STrack.

        Args:
            new_track (List[Any]): List of new STracks.
            frame_id (int): ID for current frame.
            new_id (bool, optional): New track ID. Defaults to False.
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(
        self, new_track: List[Any], frame_id: int, update_feature: bool = True
    ) -> None:
        """Update a matched track.

        Args:
            new_track (List[Any]): New STrack.
            frame_id (int): Frame ID.
            update_feature (bool, optional): Update feature. Defaults to True.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def tlwh(self) -> np.ndarray:
        """Represents a bounding box detection in a single image in
        format `(x, y, w, h)`.

        Returns:
            np.ndarray: `(min x, min y, width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Represents a bounding box detection in a single image in
        format `(x1, y1, x2, y2)`.

        Returns:
            np.ndarray: `(min x, min y, max x, max y)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: torch.Tensor) -> np.ndarray:
        """Converts format of bounding box.

        Args:
            tlwh (torch.Tensor): Bounding box with format
                `(min x, min y, width, height)`.

        Returns:
            np.ndarray: `(center x, center y, aspect ratio, height)`,
                where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Converts format of bounding box.

        Returns:
            np.ndarray: `(center x, center y, aspect ratio, height)`,
                where the aspect ratio is `width / height`.
        """
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr: torch.Tensor) -> np.ndarray:
        """Converts format of bounding box.

        Args:
            tlbr (torch.Tensor): `(min x, min y, max x, max y)`.

        Returns:
            np.ndarray: `(min x, min y, width, height)`.
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: torch.Tensor) -> np.ndarray:
        """Converts format of bounding box.

        Args:
            tlwh (torch.Tensor): `(min x, min y, width, height)`.

        Returns:
            np.ndarray: `(min x, min y, max x, max y)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self) -> str:
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches, too-many-statements, too-few-public-methods
class JDETracker:
    """JDE Tracker class."""

    def __init__(
        self, opt: Dict[str, Any], device: torch.device, frame_rate: int = 30
    ) -> None:
        self.opt = opt
        self.model = Darknet(opt["cfg"], nID=14455)
        self.model.load_state_dict(
            torch.load(opt["weights_path"], map_location="cpu")["model"], strict=False
        )
        self.model.eval().to(device)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.det_thresh = opt["score_threshold"]
        self.buffer_size = int(frame_rate / 30.0 * opt["track_buffer"])
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

    def update(self, im_blob: torch.Tensor, img0: np.ndarray) -> List[STrack]:
        """Processes the image frame and finds bounding box (detections).
        Associates the detection with corresponding tracklets and also
        handles lost, removed, refound and active tracklets.

        Args:
            im_blob (torch.Tensor): Tensor of shape depending upon the
                size of image. By default, shape of this tensor is [1, 3, 608, 1088].
            img0 (np.ndarray): ndarray of shape depending on the input
                image sequence. By default, shape is [608, 1080, 3].

        Returns:
            List[STrack]: The list contains instance information regarding
                the online_tracklets for the recieved image tensor.
        """
        self.frame_id += 1
        activated_starcks = []  # For storing active tracks, for the current frame
        refind_stracks = []
        # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []
        # The tracks which are not obtained in the current frame but are
        # not removed. (Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        # Step 1: Network forward, get detections & embeddings
        with torch.no_grad():
            pred = self.model(im_blob)
        # pred is tensor of all the proposals (default number of proposals: 54264).
        # Proposals have information associated with the bounding box and embeddings.
        pred = pred[pred[:, :, 4] > self.opt["score_threshold"]]
        # pred now has lesser number of proposals.
        # Proposals rejected on basis of object confidence score.
        if len(pred) > 0:
            dets = non_max_suppression(
                pred.unsqueeze(0),
                self.opt["score_threshold"],
                self.opt["nms_threshold"],
            )[
                0
            ].cpu()  # type: ignore
            # Final proposals are obtained in dets. Information of bounding box
            # and embeddings also included. Next step changes the detection scales.
            scale_coords(self.opt["img_size"], dets[:, :4], img0.shape).round()
            # detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred).
            # class_pred is the embeddings.
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30)
                for (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])
            ]
        else:
            detections = []

        # Add newly detected tracklets to tracked_stracks.
        unconfirmed = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                # Previous tracks which are not active in the current
                # frame are added in unconfirmed list.
                unconfirmed.append(track)
            else:
                # Active tracks are added to the local list 'tracked_stracks'.
                tracked_stracks.append(track)

        # Step 2: First association, with embedding.
        # Combining currently tracked_stracks and lost_stracks.
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with Kalman Filter.
        STrack.multi_predict(strack_pool, self.kalman_filter)

        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the
        # tracks in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the
        # detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not
                # active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        # Step 3: Second association, with IOU.
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections.
        r_tracked_stracks = []
        # This is container for stracks which were tracked till the previous
        # frame but no detection was found for it in the current frame.
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with
        # corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now
        # considering IOU_distance as measure

        for itrack in u_track:
            track = r_tracked_stracks[itrack]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks
        # are added to lost_tracks list and are marked lost.

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame.
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched.
        for itrack in u_unconfirmed:
            track = unconfirmed[itrack]
            track.mark_removed()
            removed_stracks.append(track)

        # After all these confirmation steps, if a new detection is found,
        # it is initialized for a new track.
        # Step 4: Init new stracks.
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # Step 5: Update state.
        # If the tracks are lost for more frames than the threshold number,
        # the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update the self.tracked_stracks and self.lost_stracks using the
        # updates in this step.
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # Get scores of lost tracks.
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(stracksa: List[STrack], stracksb: List[STrack]) -> List[STrack]:
    """Combines two lists of STracks together.

    Args:
        stracksa (List[STrack]): List of STracks.
        stracksb (List[STrack]): List of STracks.

    Returns:
        List[STrack]: List of joined STracks.
    """
    exists = {}
    res = []
    for track in stracksa:
        exists[track.track_id] = 1
        res.append(track)
    for track in stracksb:
        tid = track.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(track)
    return res


def sub_stracks(stracksa: List[STrack], stracksb: List[STrack]) -> List[STrack]:
    """Removes stracksb from stracksa.

    Args:
        stracksa (List[STrack]): List of STracks.
        stracksb (List[STrack]): List of STracks.

    Returns:
        List[STrack]: List of STracks.
    """
    stracks = {}
    for track in stracksa:
        stracks[track.track_id] = track
    for track in stracksb:
        tid = track.track_id
        if stracks.get(tid, 0) != 0:
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(
    stracksa: List[STrack], stracksb: List[STrack]
) -> Tuple[List[STrack], List[STrack]]:
    """Removes duplicated STracks.

    Args:
        stracksa (List[STrack]): List of STracks.
        stracksb (List[STrack]): List of STracks.

    Returns:
        Tuple[List[STrack], List[STrack]]: Lists of STracks.
    """
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p_dist, q_dist in zip(*pairs):
        timep = stracksa[p_dist].frame_id - stracksa[p_dist].start_frame
        timeq = stracksb[q_dist].frame_id - stracksb[q_dist].start_frame
        if timep > timeq:
            dupb.append(q_dist)
        else:
            dupa.append(p_dist)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
