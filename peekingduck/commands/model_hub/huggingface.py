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

"""PeekingDuck CLI `model-hub huggingface` command."""

import cmd
import logging
import shutil

import click

from peekingduck.commands import LOGGER_NAME
from peekingduck.commands.base import AliasedGroup
from peekingduck.commands.model_hub import model_hub
from peekingduck.nodes.model.huggingface_hubv1.api_utils import (
    SUPPORTED_TASKS,
    get_valid_models,
)

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name
# Replaces underscores with spaces to match expected user input
PRETTY_SUPPORTED_TASKS = [task.replace("_", " ") for task in SUPPORTED_TASKS]


@model_hub.command(cls=AliasedGroup, aliases=["hf"])
def huggingface() -> None:
    """Hugging Face Hub models."""


@huggingface.command(
    aliases=["detect", "di"],
    short_help="Returns the URL to the model's config.json which contains the "
    "detect ID-to-label mapping.",
)
@click.option(
    "--model_type",
    help=(
        "The model identifier in the format of <organization>/<model name>, "
        "e.g., facebook/detr-resnet-50."
    ),
)
def detect_ids(model_type: str) -> None:
    """Returns the URL to the model's config.json which contains the detect
    ID-to-label mapping.
    """
    url_prefix = "https://huggingface.co/"
    url_suffix = "/blob/main/config.json"
    if any(model_type in get_valid_models(task) for task in SUPPORTED_TASKS):
        print(
            f"The detect ID-to-label mapping for `{model_type}` can be found at "
            f"{url_prefix}{model_type}{url_suffix} under the `id2label` key."
        )
    else:
        print(f"{model_type} is either invalid or belongs to an unsupported task.")


@huggingface.command(
    aliases=["m"],
    short_help="Lists the valid/supported Hugging Face Hub models for the "
    "specified computer vision `task`.",
)
@click.option(
    "-t",
    "--task",
    help=f"The computer vision task, one of {PRETTY_SUPPORTED_TASKS}.",
    required=True,
)
def models(task: str) -> None:
    """Lists the valid/supported Hugging Face Hub models for the specified
    computer vision `task`.
    """
    task_code = "_".join(task.lower().split())
    if task_code in SUPPORTED_TASKS:
        valid_models = sorted(list(get_valid_models(task_code)))

        command = cmd.Cmd()
        print(f"Supported Hugging Face `{task_code}` models:")
        command.columnize(valid_models, displaywidth=shutil.get_terminal_size().columns)
    else:
        print(f"{task} is an invalid/unsupported task.")


@huggingface.command(aliases=["t"])
def tasks() -> None:
    """Lists the supported computer vision tasks."""
    print(f"Supported computer vision tasks: {PRETTY_SUPPORTED_TASKS}")
