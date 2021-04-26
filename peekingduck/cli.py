import os
import logging
import yaml
import click


from peekingduck.runner import Runner
from peekingduck.loaders import ConfigLoader, DeclarativeLoader
from peekingduck.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """
    PeekingDuck is a python framework for dealing with Machine Learning model inferences.

    Developed by Computer Vision Hub at AI Singapore.
    """
    pass


def _get_cwd():
    return os.getcwd()


def create_custom_folder():
    curdir = _get_cwd()
    custom_node_dir = os.path.join(curdir, "src/custom_nodes")
    logger.info(f"Creating custom nodes folder in {custom_node_dir}")
    os.makedirs(custom_node_dir, exist_ok=True)


def create_yml():
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
def init():
    """Initialise a PeekingDuck project"""
    print("Welcome to PeekingDuck!")
    create_custom_folder()
    create_yml()


@cli.command()
@click.option('--config_path', default=None, type=click.Path(),
              help="List of nodes to run. If none, assumes a run_config.yml at current working directory")
def run(config_path):
    """Runs PeekingDuck"""
    curdir = _get_cwd()
    if not config_path:
        config_path = os.path.join(curdir, "run_config.yml")

    custom_node_path = os.path.join(curdir, 'src/custom_nodes')

    runner = Runner(config_path, custom_node_path)
    runner.run()


@cli.command()
@click.option('--run_config_path', default='./run_config.yml', type=click.Path(),
              help="List of nodes to pull config ymls from. If none, assumes a run_config.yml at current working directory")
def get_configs(run_config_path):
    """Creates node specific config ymls for usage. If no configs are specified, pull all"""
    node_configs = ConfigLoader()

    node_loader = DeclarativeLoader(node_configs, run_config_path)
    node_loader.compile_configrc()
