# Copyright 2021 AI Singapore
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

"""
CLI functions for Peekingduck
"""

import os
import logging
import yaml
import click

from peekingduck.runner import Runner
from peekingduck.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@click.group()
def cli() -> None:
    """
    PeekingDuck is a python framework for dealing with Machine Learning model inferences.

    Developed by Computer Vision Hub at AI Singapore.
    """


def _get_cwd() -> str:
    return os.getcwd()


def create_custom_folder(custom_folder_name: str) -> None:
    """Make custom nodes folder to create custom nodes
    """
    curdir = _get_cwd()
    custom_folder_dir = os.path.join(curdir, "src", custom_folder_name)

    logger.info("Creating custom nodes folder in %s", custom_folder_dir)
    os.makedirs(custom_folder_dir, exist_ok=True)


def create_yml() -> None:
    """Inits the declarative yaml"""
    # Default yml to be discussed
    default_yml = dict(
        nodes=[
            'input.live',
            'model.yolo',
            'draw.bbox',
            'output.screen'
        ]
    )

    with open('run_config.yml', 'w') as yml_file:
        yaml.dump(default_yml, yml_file, default_flow_style=False)


@cli.command()
@click.option('--custom_folder_name', default='custom_nodes')
def init(custom_folder_name: str) -> None:
    """Initialise a PeekingDuck project"""
    print("Welcome to PeekingDuck!")
    create_custom_folder(custom_folder_name)
    create_yml()


@cli.command()
@click.option('--config_path', default=None, type=click.Path(),

              help="List of nodes to run. None assumes \
                   run_config.yml at current working directory")
def run(config_path: str) -> None:
    """Runs PeekingDuck"""
    curdir = _get_cwd()
    if not config_path:
        config_path = os.path.join(curdir, "run_config.yml")

    runner = Runner(config_path, "src")
    runner.run()
