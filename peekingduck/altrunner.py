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
Main engine for Peekingduck processes
"""


import time
import logging
import importlib
from typing import List, Callable
# from multiprocessing import Process, Manager
from multiprocess import Process, Manager



class Runner():
    """Runner class that uses the declared nodes to create pipeline to run inference
    """

    def __init__(self,
                 nodes):
        """
        """
        self.nodes = nodes
        self.logger = logging.getLogger(__name__)

    def _get_process(self, node_name) -> Callable:
        
        # implementation of the nodes
        def node_runner(shared_data):
            Node = importlib.import_module(node_name)
            node = Node.Node()
            inputs_needed = node.input

            # p1-live: img1, img2
            # p3-draw: img1*, img2*
            # p4-out : 

            # take required data and run
            # {"img": [img1, img2, img3]}
            #  img1, img2, img3, img4
            #Y:img1, ...   ... , img4          ...     ... 
            #D: ...  ...   ... , BBimg1+img4                 BBimg4+img7
            #O:img1, img2, img3, BBimg1+img4   BBimg1+img5 
            while True: # when live cannot get frame img:None
                if inputs_needed[0] == "none" or (inputs_needed[0] in shared_data and (shared_data[inputs_needed[0]] is not None)):
                    # retrieve next required data
                    results = node.run(shared_data)
                    shared_data.update(results)
                    # add results into queue
                    # check for last usage of data and delete when used
    
        return node_runner

    def run(self):
        # create data structure
        # pass datastructure as args
        shared_dict = Manager().dict()

        processes = [Process(target=self._get_process(node), args=(shared_dict,)) for node in self.nodes]


        for process in processes:
            process.start()
            print("I am trying to start the next process")
        
        for process in processes:
            process.join()