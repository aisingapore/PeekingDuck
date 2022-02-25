import numpy as np
import torch

from peekingduck.pipeline.nodes.model.fairmotv1.fairmot_files import tracker
from peekingduck.pipeline.nodes.model.fairmotv1.fairmot_files.track import STrack


class TestFairMOTTracker:
    def test_remove_duplicate_stracks(self):
        tlwhs = [
            np.array([10, 20, 30, 40]),
            np.array([20, 40, 60, 80]),
            np.array([40, 80, 120, 160]),
        ]
        score = torch.ones((1,), dtype=torch.float, device=torch.device("cpu"))
        feature = np.ones((10,), dtype=float)
        # Create 2 list of elementwise overlapping stracks
        stracks_1 = [STrack(tlwh, score, feature) for tlwh in tlwhs]
        stracks_2 = [STrack(tlwh, score, feature) for tlwh in tlwhs]

        frame_id = 10
        early_start = 2
        late_start = 5
        for i, _ in enumerate(tlwhs):
            stracks_1[i].frame_id = frame_id
            stracks_2[i].frame_id = frame_id
            # Alternate between which STrack is older so we cover more branches
            if i % 2 == 0:
                stracks_1[i].start_frame = early_start
                stracks_2[i].start_frame = late_start
            else:
                stracks_1[i].start_frame = late_start
                stracks_2[i].start_frame = early_start
        out_stracks_1, out_stracks_2 = tracker._remove_duplicate_stracks(
            stracks_1, stracks_2
        )
        # stracks_1 has more older tracks so more elements will be leftover
        assert len(out_stracks_1) == 2
        assert len(out_stracks_2) == 1
