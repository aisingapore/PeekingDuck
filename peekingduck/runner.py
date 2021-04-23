import os
import sys
import yaml
import importlib
import logging

from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.config import ConfigLoader
END_TYPE = 'process_end'

"""
Combine runner at this level. Use this to create the graph, and waddle is loop or once
"""

class Runner():
    """Runner class that uses the declared nodes to create pipeline to run inference
    """

    def __init__(self, RUN_PATH, CUSTOM_NODE_PATH=None, nodes=[]):
        """
        Args:
            RUN_PATH (str): path to yaml file of node pipeine declaration.
            CUSTOM_NODE_PATH (str): path to custom nodes folder if used.
            nodes (:obj:'list' of :obj:'Node'): if not using declarations via yaml,
                initialize by giving the node stack as a list

        """

        self.logger = logging.getLogger(__name__)

        instantiated_nodes = []

        if not nodes:
            with open(RUN_PATH) as file:
                run_config = yaml.load(file, Loader=yaml.FullLoader)
                self.logger.info(
                    'Successfully loaded run_config file. Proceeding to create Graph.')
            # create Graph to run
            nodes_config = ConfigLoader(run_config['nodes'])
            imported_nodes = []
            for node in run_config['nodes']:
                if node.split('.')[0] == 'custom':
                    custom_node_path = os.path.join(
                        CUSTOM_NODE_PATH, node.split('.')[1]+'.py')
                    spec = importlib.util.spec_from_file_location(
                        node, custom_node_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    imported_nodes.append(("custom", module))
                else:
                    imported_nodes.append((node, importlib.import_module(
                        'peekingduck.pipeline.nodes.' + node)))
                self.logger.info("{} added to pipeline.".format(node))

            # instantiate classes from imported nodes
            for node_name, node in imported_nodes:
                if node_name == 'custom':
                    instantiated_nodes.append(node.Node(None))
                else:
                    config = nodes_config.get(node_name)
                    instantiated_nodes.append(node.Node(config))

        # If Runner given nodes, instantiated_nodes is created differently
        else:
            for node in nodes:
                instantiated_nodes.append(node)

        # Create Graph
        try:
            self.pipeline = Pipeline(instantiated_nodes)
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(1)

    def run(self):
        """execute single or continuous inference
        """
        while not self.pipeline.video_end:
            self.pipeline.execute()
