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

from peekingduck.commands.model_hub import model_hub
from peekingduck.pipeline.nodes.model.mediapipe_hubv1.api_doc import SUPPORTED_TASKS

logger = logging.getLogger("peekingduck.cli")  # pylint: disable=invalid-name


@model_hub.group()
def mediapipe() -> None:
    """Utility commands for MediaPipe models."""


@mediapipe.command()
@click.option("--task", help="Computer vision task, e.g., object detection")
@click.option("--subtask", help="Computer vision subtask, e.g., face")
def model_types(task: str, subtask: str) -> None:
    """Lists the supported model types for the specified `task` and `subtask`."""
    print(f"Supported model types for '{task}/{subtask}'")
    for model_type, docstring in SUPPORTED_TASKS.get_model_cards(
        _unprettify(task), _unprettify(subtask)
    ).items():
        print(f"{model_type}: {docstring}")


@mediapipe.command()
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
