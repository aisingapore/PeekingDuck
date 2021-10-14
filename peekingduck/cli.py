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
from peekingduck.utils.logger import LoggerSetup
from peekingduck import __version__

LoggerSetup()
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """
    PeekingDuck is a modular computer vision inference framework.

    Developed by Computer Vision Hub at AI Singapore.
    """


def _get_cwd() -> str:
    return os.getcwd()


def create_custom_folder(custom_folder_name: str) -> None:
    """Make custom nodes folder to create custom nodes
    """
    curdir = _get_cwd()
    custom_folder_dir = os.path.join(curdir, "src", custom_folder_name)
    custom_configs_dir = os.path.join(custom_folder_dir, 'configs')

    logger.info("Creating custom nodes folder in %s", custom_folder_dir)
    os.makedirs(custom_folder_dir, exist_ok=True)
    os.makedirs(custom_configs_dir, exist_ok=True)


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
@click.option('--node_config', default="None",
              help="""Modify node configs by wrapping desired configs in a JSON string.\n
                    Example: --node_config '{"node_name": {"param_1": var_1}}' """)
def run(config_path: str, node_config: str) -> None:
    """Runs PeekingDuck"""

    curdir = _get_cwd()
    if not config_path:
        config_path = os.path.join(curdir, "run_config.yml")

    runner = Runner(config_path, node_config, "src")
    runner.run()


@cli.command()
@click.argument("type_name", required=False)
def nodes(type_name: str = None) -> None:
    """
    Lists available nodes in PeekingDuck. When no argument is given, all
    available nodes will be listed. When the node type is given as an argument
    , all available nodes in the specified node type will be listed.

    Args:
        type_name (str): input, model, dabble, draw or output
    """

    url_prefix = "https://peekingduck.readthedocs.io/en/stable/peekingduck.pipeline.nodes."
    url_postfix = ".Node.html#peekingduck.pipeline.nodes."

    if type_name is None:
        type_of_node = ["input", "model", "dabble", "draw", "output"]
    else:
        type_of_node = [type_name]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    configs_dir = os.path.join(dir_path, 'configs')

    for node_type in type_of_node:
        type_dir = os.path.join(configs_dir, node_type)
        files_name = os.listdir(type_dir)

        click.secho("\nPeekingDuck has the following ", bold=True, nl=False)
        click.secho(f"{node_type} ", fg="red", bold=True, nl=False)
        click.secho("nodes:", bold=True)

        node_names = [file_name.split(".")[0] for file_name in files_name]
        max_length = len(max(node_names, key=len))
        for num, node_name in enumerate(node_names):
            url = (
                f" {url_prefix}{node_type}.{node_name}{url_postfix}"
                f"{node_type}.{node_name}.Node"
            )

            click.secho(f"{num + 1}:", nl=False)
            click.secho(f"{node_name: <{max_length + 1}}", bold=True, nl=False)
            click.secho("Info:", fg="yellow", nl=False)
            click.secho(url)

    click.secho("\n")
