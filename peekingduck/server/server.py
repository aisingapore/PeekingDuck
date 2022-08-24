# Copyright 2022 AI Singapore
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
Implement PeekingDuck Server
"""


import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List

from fastapi import FastAPI, Body
import pika
import uvicorn

from peekingduck.declarative_loader import DeclarativeLoader, NodeList
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.utils.requirement_checker import RequirementChecker


EXCHANGE_TYPE = "fanout"


class Server:
    """Implement PeekingDuck Server class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
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

    def get_pipeline(self) -> NodeList:
        """Retrieves run configuration.

        Returns:
            (:obj:`Dict`): Run configurations being used by runner.
        """
        return self.node_loader.node_list

    def process_nodes(self, body: Dict[str, Any]) -> None:
        """Process nodes in PeekingDuck pipeline."""

        for node in self.pipeline.nodes:
            if self.pipeline.data.get("pipeline_end", False):
                self.pipeline.terminate = True
                if "pipeline_end" not in node.inputs:
                    continue

            if "all" in node.inputs:
                inputs = copy.deepcopy(self.pipeline.data)
            elif "message" in node.inputs:
                inputs = {"message": body}
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


class ReqRes(Server):
    """Implement PeekingDuck ReqRes class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        nodes: List[AbstractNode] = None,
        host: str = None,
        port: int = None,
    ) -> None:
        super().__init__(
            pipeline_path, config_updates_cli, custom_nodes_parent_subdir, nodes
        )
        self.app = FastAPI()
        self.host = host
        self.port = port

    def run(self) -> None:  # pylint: disable=too-many-branches
        """execute single or continuous inference"""

        @self.app.post("/")
        async def receive(body: dict = Body) -> None:  # type: ignore # pylint: disable=unused-variable
            self.process_nodes(body)
            return

        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

        # clean up nodes with threads
        for node in self.pipeline.nodes:
            if node.name.endswith(".visual"):
                node.release_resources()


class PubSub(Server):
    """Implement PeekingDuck PubSub class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        nodes: List[AbstractNode] = None,
        host: str = None,
        username: str = None,
        password: str = None,
        exchange_name: str = None,
    ) -> None:
        super().__init__(
            pipeline_path, config_updates_cli, custom_nodes_parent_subdir, nodes
        )
        credentials = pika.PlainCredentials(username=username, password=password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, credentials=credentials)
        )
        self.channel = connection.channel()
        self.channel.exchange_declare(
            exchange=exchange_name, exchange_type=EXCHANGE_TYPE
        )

        # use random queue name, and "exclusive" flag will delete queue after consumer disconnects
        result = self.channel.queue_declare(queue="", exclusive=True)
        self.queue_name = result.method.queue
        # To tell the exchange to send messages to the queue
        self.channel.queue_bind(exchange=exchange_name, queue=self.queue_name)

    def run(self) -> None:  # pylint: disable=too-many-branches
        """execute single or continuous inference"""
        self.channel.basic_consume(
            queue=self.queue_name, on_message_callback=self._callback, auto_ack=True
        )
        self.channel.start_consuming()

        # clean up nodes with threads
        for node in self.pipeline.nodes:
            if node.name.endswith(".visual"):
                node.release_resources()

    def _callback(  # pylint: disable=unused-argument
        self, channel: Callable, method: Callable, props: Callable, body: bytes
    ) -> None:
        body_dict = json.loads(body)
        self.process_nodes(body_dict)


class Queue(Server):
    """Implement PeekingDuck Queue class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        nodes: List[AbstractNode] = None,
        host: str = None,
        username: str = None,
        password: str = None,
        queue_name: str = None,
    ) -> None:
        super().__init__(
            pipeline_path, config_updates_cli, custom_nodes_parent_subdir, nodes
        )
        self.queue_name = queue_name
        credentials = pika.PlainCredentials(username=username, password=password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, credentials=credentials)
        )
        self.channel = connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)

    def run(self) -> None:  # pylint: disable=too-many-branches
        """execute single or continuous inference"""

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name, on_message_callback=self._callback
        )
        self.channel.start_consuming()

        # clean up nodes with threads
        for node in self.pipeline.nodes:
            if node.name.endswith(".visual"):
                node.release_resources()

    def _callback(  # pylint: disable=unused-argument
        self, channel: Callable, method: Callable, props: Callable, body: bytes
    ) -> None:
        body_dict = json.loads(body)
        self.process_nodes(body_dict)
        channel.basic_ack(delivery_tag=method.delivery_tag)
