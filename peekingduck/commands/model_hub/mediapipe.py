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

"""PeekingDuck CLI `model-hub mediapipe` command."""

import logging

import click

from peekingduck.commands import LOGGER_NAME
from peekingduck.commands.base import AliasedGroup
from peekingduck.commands.model_hub import model_hub
from peekingduck.nodes.model.mediapipe_hubv1.api_doc import SUPPORTED_TASKS

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name

subtask_option = click.option(
    "-s", "--subtask", help="Computer vision subtask, e.g., face, body.", required=True
)


@model_hub.command(cls=AliasedGroup, aliases=["mp"])
def mediapipe() -> None:
    """MediaPipe Solutions models."""


@mediapipe.command(
    aliases=["keypoint", "kf"],
    short_help=(
        "Lists the supported keypoint formats for the specified pose "
        "estimation `subtask`."
    ),
)
@subtask_option
def keypoint_formats(subtask: str) -> None:
    """Lists the supported keypoint formats for the specified pose estimation `subtask`."""
    print(f"Supported keypoint formats for 'pose estimation/{subtask}'")
    for keypoint_format, docstring in SUPPORTED_TASKS.get_keypoint_format_cards(
        _unprettify(subtask)
    ).items():
        print(f"{keypoint_format}: {docstring}")


@mediapipe.command(
    aliases=["model", "mt"],
    short_help="Lists the supported model types for the specified `task` and `subtask`.",
)
@click.option(
    "-t", "--task", help="Computer vision task, e.g., object detection", required=True
)
@subtask_option
def model_types(task: str, subtask: str) -> None:
    """Lists the supported model types for the specified `task` and `subtask`."""
    print(f"Supported model types for '{task}/{subtask}'")
    for model_type, docstring in SUPPORTED_TASKS.get_model_cards(
        _unprettify(task), _unprettify(subtask)
    ).items():
        print(f"{model_type}: {docstring}")


@mediapipe.command(aliases=["t"])
def tasks() -> None:
    """Lists the supported computer vision tasks."""
    print("Supported computer vision tasks and respective subtasks:")
    for task in SUPPORTED_TASKS.tasks:
        print(_prettify(task))
        for subtask in SUPPORTED_TASKS.get_subtasks(task):
            print(f"\t{_prettify(subtask)}")


def _prettify(item: str) -> str:
    """Replaces underscores with spaces to match expected user input."""
    return item.replace("_", " ")


def _unprettify(item: str) -> str:
    """Replaces spaces with underscores to match stored task and subtask keys."""
    return item.replace(" ", "_")
