import os
import yaml
import click

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
    print (f"Creating custom nodes folder in {custom_node_dir}")
    os.makedirs(custom_node_dir, exist_ok = True)


def create_yml():
    """Inits the declarative yaml"""
    #Default yml to be discussed
    default_yml = dict(
        nodes = [
            'input.live',
            'model.yolo',
            'draw.bbox',
            'output.screen'
        ]
    )

    with open('run_config.yml', 'w') as yml_file:
        yaml.dump(default_yml, yml_file, default_flow_style = False)

@cli.command()
@click.option('--config_path', default = None, type=click.Path(),
              help="List of nodes to pull config ymls from. If none, assumes a run_config.yml at project root")
def get_configs(config_path):
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
            module, node_name = node.split('.')
            if node_path[0] == 'custom':
                node_config_path = os.path.join('src/custom_nodes', node_name, 'config.yml')
            else:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                config_filename =  module + '_' + node_name + '.yml'
                node_config_path = os.path.join(dir_path, 'configs', config_filename)
            if os.path.isfile(node_config_path):
                with open(node_config_path, 'r') as node_yml:
                    node_config = yaml.load(node_yml, Loader=yaml.FullLoader)
                    node_config = {node_name: node_config}
                yaml.dump(node_config, node_configs, default_flow_style = False)
            else:
                print(f'No associated configs found for {node}. Skipping')


@cli.command()
def init():
    """Initialise a PeekingDuck project"""
    print("Welcome to PeekingDuck!")
    create_custom_folder()
    create_yml()