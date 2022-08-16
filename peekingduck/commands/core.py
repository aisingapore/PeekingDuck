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

"""Core PeekingDuck CLI commands."""

import logging
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Union

import click
import yaml

from peekingduck.commands import LOGGER_NAME
from peekingduck.runner import Runner
from peekingduck.utils.deprecation import deprecate
from peekingduck.utils.logger import LoggerSetup
from peekingduck.viewer import Viewer

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


@click.command()
@click.option("--custom_folder_name", default="custom_nodes")
def init(custom_folder_name: str) -> None:
    """Initializes a PeekingDuck project."""
    print("Welcome to PeekingDuck!")
    _create_custom_folder(custom_folder_name)
    _create_pipeline_config_yml()


@click.command()
@click.option(
    "--config_path",
    default=None,
    type=click.Path(),
    help=(
        "List of nodes to run. None assumes pipeline_config.yml at current working directory"
    ),
)
@click.option(
    "--log_level",
    default="info",
    help="""Modify log level {"critical", "error", "warning", "info", "debug"}""",
)
@click.option(
    "--node_config",
    default="None",
    help="""Modify node configs by wrapping desired configs in a JSON string.\n
        Example: --node_config '{"node_name": {"param_1": var_1}}'""",
)
@click.option(
    "--num_iter",
    default=None,
    type=int,
    help="Stop pipeline after running this number of iterations",
)
@click.option(
    "--viewer",
    default=False,
    is_flag=True,
    help="Launch PeekingDuck viewer",
)
def run(  # pylint: disable=too-many-arguments
    config_path: str,
    log_level: str,
    node_config: str,
    num_iter: int,
    viewer: bool,
    nodes_parent_dir: str = "src",
) -> None:
    """Runs PeekingDuck"""
    LoggerSetup.set_log_level(log_level)

    if config_path is None:
        curr_dir = Path.cwd()
        if (curr_dir / "pipeline_config.yml").is_file():
            config_path = curr_dir / "pipeline_config.yml"
        elif (curr_dir / "run_config.yml").is_file():
            deprecate(
                "using 'run_config.yml' as the default pipeline configuration "
                "file is deprecated and will be removed in the future. Please "
                "use 'pipeline_config.yml' instead.",
                2,
            )
            config_path = curr_dir / "run_config.yml"
        else:
            config_path = curr_dir / "pipeline_config.yml"
    pipeline_config_path = Path(config_path)

    if viewer:
        logger.info("Launching PeekingDuck Viewer")
        start_time = perf_counter()
        pkd_viewer = Viewer(
            pipeline_path=pipeline_config_path,
            config_updates_cli=node_config,
            custom_nodes_parent_subdir=nodes_parent_dir,
            num_iter=num_iter,
        )
        end_time = perf_counter()
        logger.debug(f"Startup time = {end_time - start_time:.2f} sec")
        pkd_viewer.run()
    else:
        start_time = perf_counter()
        runner = Runner(
            pipeline_path=pipeline_config_path,
            config_updates_cli=node_config,
            custom_nodes_parent_subdir=nodes_parent_dir,
            num_iter=num_iter,
        )
        end_time = perf_counter()
        logger.debug(f"Startup time = {end_time - start_time:.2f} sec")
        runner.run()


@click.command()
def verify_install() -> None:
    """Verifies PeekingDuck installation by running object detection on
    'wave.mp4'.
    """
    LoggerSetup.set_log_level("info")

    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline_config_path = Path(tmp_dir) / "verification_pipeline.yml"

        _create_pipeline_config_yml(
            [
                {
                    "input.visual": {
                        "source": "https://storage.googleapis.com/peekingduck/videos/wave.mp4"
                    }
                },
                "model.yolo",
                "draw.bbox",
                "output.screen",
            ],
            pipeline_config_path,
        )

        runner = Runner(
            pipeline_path=pipeline_config_path,
            config_updates_cli="None",
            custom_nodes_parent_subdir="src",
            num_iter=None,
        )
        runner.run()
        os.chdir(cwd)


def _create_custom_folder(custom_folder_name: str) -> None:
    """Makes custom nodes folder to create custom nodes.

    Args:
        custom_folder_name (:obj:`str`): Name of the custom nodes folder.
    """
    curr_dir = Path.cwd()
    custom_nodes_dir = curr_dir / "src" / custom_folder_name
    custom_nodes_config_dir = custom_nodes_dir / "configs"

    logger.info(f"Creating custom nodes folder in {custom_nodes_dir}")
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    custom_nodes_config_dir.mkdir(parents=True, exist_ok=True)


def _create_pipeline_config_yml(
    default_nodes: List[Union[str, Dict[str, Any]]] = None,
    default_path: Path = Path("pipeline_config.yml"),
) -> None:
    """Initializes the declarative *pipeline_config.yml*.

    Args:
        default_nodes (List[Union[str, Dict[str, Any]]]): A list of PeekingDuck
            node. For nodes with custom configuration, use a dictionary instead
            of a string.
        default_path (Path): Path of the pipeline config file.
    """
    # Default yml to be discussed
    if default_nodes is None:
        default_nodes = [
            {
                "input.visual": {
                    "source": "https://storage.googleapis.com/peekingduck/videos/wave.mp4"
                }
            },
            "model.posenet",
            "draw.poses",
            "output.screen",
        ]
    default_yml = dict(nodes=default_nodes)

    with open(default_path, "w") as yml_file:
        yaml.dump(default_yml, yml_file, default_flow_style=False)
