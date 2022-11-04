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

"""Util functions to provide documentation for MediaPipe API."""

from typing import Dict, Set

_TASK_SUBTASK_MODEL_TYPE = {
    "object_detection": {
        "face": {
            0: (
                "A short-range model that works best for faces within 2 meters "
                "from the camera."
            ),
            1: "A full-range model best for faces within 5 meters.",
        }
    },
    "pose_estimation": {
        "body": {
            0: "BlazePose GHUM Lite, lower inference latency at the cost of landmark accuracy.",
            1: "BlazePose GHUM Full.",
            2: "BlazePose GHUM Heavy, higher landmark accuracy at the cost of inference latency.",
        },
        "hand": {
            0: "Palm detection model (lite) + Hand landmark model.",
            1: (
                "Palm detection model (full) + Hand landmark model, higher landmark "
                "accuracy at the cost of inference latency."
            ),
        },
    },
}

_KEYPOINT_FORMAT = {
    "body": {
        "blaze_pose": "Keypoint format used by the BlazePose model. Contains 33 keypoints.",
        "coco": "Keypoint format used by COCO dataset. Contains 17 keypoints.",
    },
    "hand": {
        "coco": "Keypoint format used by COCO-WholeBody dataset. Contains 21 keypoints.",
    },
}


class SupportedTasks:
    """Utility class to extract information from task->subtask->model type
    mapping of MediaPipe models.
    """

    def __init__(
        self,
        task_subtask_model_type: Dict[str, Dict[str, Dict[int, str]]],
        keypoint_format: Dict[str, Dict[str, str]],
    ) -> None:
        self.task_subtask_model_type = task_subtask_model_type
        self.keypoint_format = keypoint_format

    @property
    def tasks(self) -> Set[str]:
        """Supported computer vision tasks."""
        return set(self.task_subtask_model_type.keys())

    def get_keypoint_formats(self, subtask: str) -> Set[str]:
        """Supported keypoint formats."""
        return set(self.keypoint_format[subtask].keys())

    def get_keypoint_format_cards(self, subtask: str) -> Dict[str, str]:
        """Supported keypoint formats and details for the specified `subtask`."""
        return self.keypoint_format[subtask]

    def get_model_cards(self, task: str, subtask: str) -> Dict[int, str]:
        """Supported model types and details for the specified computer vision
        `task` and `subtask`.
        """
        return self.task_subtask_model_type[task][subtask]

    def get_model_types(self, task: str, subtask: str) -> Set[int]:
        """Supported model types for the specified computer vision `task` and
        `subtask`.
        """
        return set(self.task_subtask_model_type[task][subtask].keys())

    def get_subtask_model_types(self, task: str) -> Dict[str, Set[int]]:
        """Subtask to model types mapping for the specified computer vision
        `task`.
        """
        return {
            subtask: self.get_model_types(task, subtask)
            for subtask in self.get_subtasks(task)
        }

    def get_subtasks(self, task: str) -> Set[str]:
        """Supported subtasks for the specified computer vision `task`."""
        return set(self.task_subtask_model_type[task].keys())


SUPPORTED_TASKS = SupportedTasks(_TASK_SUBTASK_MODEL_TYPE, _KEYPOINT_FORMAT)
