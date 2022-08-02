import copy
import logging
import sys
from pathlib import Path

# from time import perf_counter
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from peekingduck.declarative_loader import DeclarativeLoader, NodeList
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.utils.requirement_checker import RequirementChecker


app = FastAPI()

# This needs to be flexible
class Item(BaseModel):
    name: str
    image: str
    timestamp: str


class Server:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        num_iter: int = None,
        nodes: List[AbstractNode] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        try:
            if nodes:
                # instantiated_nodes is created differently when given nodes
                self.pipeline = Pipeline(nodes)
            elif pipeline_path and config_updates_cli and custom_nodes_parent_subdir:
                # create Graph to run
                self.node_loader = DeclarativeLoader(
                    pipeline_path, config_updates_cli, custom_nodes_parent_subdir
                )
                self.pipeline = self.node_loader.get_pipeline()
            else:
                raise ValueError(
                    "Arguments error! Pass in either nodes to load directly via "
                    "Pipeline or pipeline_path, config_updates_cli, and "
                    "custom_nodes_parent_subdir to load via DeclarativeLoader."
                )
        except ValueError as error:
            self.logger.error(str(error))
            sys.exit(1)
        if RequirementChecker.n_update > 0:
            self.logger.warning(
                f"{RequirementChecker.n_update} package"
                f"{'s' * int(RequirementChecker.n_update > 1)} updated. "
                "Please rerun for the updates to take effect."
            )
            sys.exit(3)
        if num_iter is None or num_iter <= 0:
            self.num_iter = 0
        else:
            self.num_iter = num_iter
            self.logger.info(f"Run pipeline for {num_iter} iterations")

    def run(self) -> None:  # pylint: disable=too-many-branches
        """execute single or continuous inference"""
        # num_iter = 0

        # while not self.pipeline.terminate:
        @app.post("/image")
        async def image(item: Item):
            for node in self.pipeline.nodes:
                # if num_iter == 0:  # report node setup times at first iteration
                #     self.logger.debug(f"First iteration: setup {node.name}...")
                #     node_start_time = perf_counter()
                if self.pipeline.data.get("pipeline_end", False):
                    self.pipeline.terminate = True
                    if "pipeline_end" not in node.inputs:
                        continue

                if "all" in node.inputs:
                    inputs = copy.deepcopy(self.pipeline.data)
                elif "request" in node.inputs:
                    inputs = {"request": item}
                else:
                    inputs = {
                        key: self.pipeline.data[key]
                        for key in node.inputs
                        if key in self.pipeline.data
                    }
                if hasattr(node, "optional_inputs"):
                    for key in node.optional_inputs:
                        # The nodes will not receive inputs with the optional
                        # key if it's not found upstream
                        if key in self.pipeline.data:
                            inputs[key] = self.pipeline.data[key]

                outputs = node.run(inputs)
                self.pipeline.data.update(outputs)
            #     if num_iter == 0:
            #         node_end_time = perf_counter()
            #         self.logger.debug(
            #             f"{node.name} setup time = {node_end_time - node_start_time:.2f} sec"
            #         )
            # num_iter += 1
            # if self.num_iter > 0 and num_iter >= self.num_iter:
            #     self.logger.info(f"Stopping pipeline after {num_iter} iterations")
            #     break
            return item.name

        # https://www.uvicorn.org/deployment/#running-programmatically
        # This doesn't work:
        # uvicorn.run("server:app", host="127.0.0.1", port=5000, log_level="info")
        # multiple workers probably not a good idea with PKD - e.g. media_writer needs to combine
        # correctly, weights downloading will have issues, etc
        uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")

        # clean up nodes with threads
        for node in self.pipeline.nodes:
            if node.name.endswith(".visual"):
                node.release_resources()

    def get_pipeline(self) -> NodeList:
        """Retrieves run configuration.

        Returns:
            (:obj:`Dict`): Run configurations being used by runner.
        """
        return self.node_loader.node_list
