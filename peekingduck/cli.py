"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import logging
import yaml
import click
from peekingduck.runner import Runner
from .utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__) #pylint: disable=invalid-name

@click.group()
def cli():
    """
    PeekingDuck is a python framework for dealing with Machine Learning model inferences.

    Developed by Computer Vision Hub at AI Singapore.
    """

def _get_cwd():
    return os.getcwd()

def create_custom_folder():
    """Make custom nodes folder to create custom nodes
    """
    curdir = _get_cwd()
    custom_node_dir = os.path.join(curdir, "src/custom_nodes")
    logger.info("Creating custom nodes folder in %s", custom_node_dir)
    os.makedirs(custom_node_dir, exist_ok=True)


def create_yml():
    """Inits the declarative yaml"""
    #Default yml to be discussed
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
def init():
    """Initialise a PeekingDuck project"""
    print("Welcome to PeekingDuck!")
    create_custom_folder()
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

    custom_node_path = os.path.join(curdir, 'src/custom_nodes')

    runner = Runner(config_path, custom_node_path)
    runner.run()


@cli.command()
@click.option('--config_path', default=None, type=click.Path(),
              help="List of nodes to pull config ymls from. \
                   If none, assumes a run_config.yml at current working directory")
def get_configs(config_path: str) -> None:
    """Creates node specific config ymls for usage. If no configs are specified, pull all"""
    if not config_path:
        curdir = _get_cwd()
        config_path = os.path.join(curdir, "run_config.yml")

    with open(config_path) as node_yml:
        nodes = yaml.load(node_yml, Loader=yaml.FullLoader)['nodes']

    if os.path.isfile('node_config.yml'):
        os.remove('node_config.yml')

    #should use ConfigLoader() here as well
    with open('node_config.yml', 'a') as node_configs:
        for node in nodes:
            node_type, node_name = node.split('.')
            if node_type == 'custom':
                node_config_path = os.path.join('src/custom_nodes', node_name, 'config.yml')
            else:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                config_filename = node_name + '.yml'
                node_config_path = os.path.join(dir_path, 'configs', node_type, config_filename)
            if os.path.isfile(node_config_path):
                with open(node_config_path, 'r') as node_yml:
                    node_config = yaml.load(node_yml, Loader=yaml.FullLoader)
                    node_config = {node_name: node_config}
                yaml.dump(node_config, node_configs, default_flow_style=False)
            else:
                logger.info('No associated configs found for %s. Skipping.', node)
