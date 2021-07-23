import os
import yaml
from typing import Any, Dict

class ConfigLoader:  # pylint: disable=too-few-public-methods
    """ Reads configuration and returns configuration required to
    create the nodes for the project
    """

    def __init__(self, basedir: str) -> None:
        self._basedir = basedir

    def _get_config_path(self, node: str) -> str:
        """ Based on the node, return the corresponding node config path """

        configs_folder = os.path.join(self._basedir, 'configs')
        node_type, node_name = node.split(".")
        node_name = node_name + ".yml"
        filepath = os.path.join(configs_folder, node_type, node_name)

        return filepath

    def get(self, node_name: str) -> Dict[str, Any]:
        """Get item from configuration read from the filepath,
        item refers to the node item configuration you are trying to get"""

        filepath = self._get_config_path(node_name)

        with open(filepath) as file:
            node_config = yaml.safe_load(file)

        # some models require the knowledge of where the root is for loading
        node_config['root'] = self._basedir
        return node_config
