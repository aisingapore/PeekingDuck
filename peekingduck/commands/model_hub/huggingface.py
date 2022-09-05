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

"""PeekingDuck CLI `model-hub` command."""

import cmd
import logging
import shutil

import click

from peekingduck.commands.model_hub import model_hub
from peekingduck.pipeline.nodes.model.huggingfacev1.api_utils import get_valid_models

logger = logging.getLogger("peekingduck.cli")  # pylint: disable=invalid-name


@model_hub.group()
def huggingface() -> None:
    """Utility commands for Hugging Face Hub models."""


@huggingface.command()
@click.option(
    "--task", help="The computer vision task, e.g. object detection.", required=True
)
def models(task: str) -> None:
    """Lists the valid/supported Hugging Face Hub models for the specified
    computer vision `task`.
    """
    try:
        task_code = "_".join(task.lower().split())
        valid_models = sorted(list(get_valid_models(task_code)))

        command = cmd.Cmd()
        print(f"Supported Hugging Face `{task_code}` models:")
        command.columnize(valid_models, displaywidth=shutil.get_terminal_size().columns)
    except KeyError:
        print(f"{task} is an invalid/unsupported task.")
