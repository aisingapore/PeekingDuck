import os
import sys
import yaml
import importlib
import logging

from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.loaders import ConfigLoader, DeclarativeLoader
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

        if not nodes:
            node_configs = ConfigLoader()
            # create Graph to run
            node_loader = DeclarativeLoader(node_configs, RUN_PATH, CUSTOM_NODE_PATH)

            self.pipeline = node_loader.get_nodes()

        # If Runner given nodes, instantiated_nodes is created differently
        else:
            try:
                self.pipeline = Pipeline(nodes)
            except ValueError as e:
                self.logger.error(str(e))
                sys.exit(1)

    def run(self):
        """execute single or continuous inference
        """
        while not self.pipeline.video_end:
            self.pipeline.execute()
