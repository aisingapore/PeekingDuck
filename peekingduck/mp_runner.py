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
Prototype engine to run individual peekingduck nodes in multiple processes
"""


import logging
import importlib

from typing import Dict, List, Callable
from multiprocess import Process, Manager  # pylint: disable=no-name-in-module


class Runner: # pylint: disable=too-few-public-methods
    """
    Runner class to create the different processes and to run the processes in parallel
    """

    def __init__(self, nodes: List[str]):
        """ """
        self.nodes = nodes
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_process(node_name: str) -> Callable:

        # implementation of the nodes
        def node_runner(shared_data: Dict) -> None:
            node = importlib.import_module(node_name).Node()
            inputs_needed = node.input

            while True:  # when live cannot get frame img:None
                if inputs_needed[0] == "none" or (
                    inputs_needed[0] in shared_data
                    and (shared_data[inputs_needed[0]] is not None)
                ):
                    # retrieve next required data
                    results = node.run(shared_data)
                    shared_data.update(results)

        def node_runner_special(shared_data: Dict) -> None:
            draw_node = importlib.import_module(
                "peekingduck.pipeline.nodes.draw.bbox"
            ).Node()
            screen_node = importlib.import_module(
                "peekingduck.pipeline.nodes.output.screen"
            ).Node()
            inputs_needed = draw_node.input

            # take required data and run
            while True:
                if inputs_needed[0] == "none" or (
                    inputs_needed[0] in shared_data
                    and (shared_data[inputs_needed[0]] is not None)
                ):
                    results = draw_node.run(shared_data)
                    shared_data.update(results)
                    results = screen_node.run(shared_data)

        if node_name == "special":
            return node_runner_special
        return node_runner

    def run(self) -> None: # pylint: disable=missing-function-docstring
        # create data structure
        # pass datastructure as args
        shared_dict = Manager().dict()

        processes = [
            Process( # pylint: disable=not-callable
                target=self._get_process(node), args=(shared_dict,)
            )
            for node in self.nodes
        ]

        for process in processes:
            process.start()
            print("I am trying to start the next process")

        for process in processes:
            process.join()
