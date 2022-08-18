import copy
import logging
import sys
from pathlib import Path

from typing import List

from fastapi import FastAPI, Body
import pickle
import pika
import uvicorn

from peekingduck.declarative_loader import DeclarativeLoader, NodeList
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.utils.requirement_checker import RequirementChecker


EXCHANGE_TYPE = "fanout"


class Server:
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

    def _process_nodes(self, item) -> None:

        for node in self.pipeline.nodes:
            if self.pipeline.data.get("pipeline_end", False):
                self.pipeline.terminate = True
                if "pipeline_end" not in node.inputs:
                    continue

            if "all" in node.inputs:
                inputs = copy.deepcopy(self.pipeline.data)
            elif "message" in node.inputs:
                inputs = {"message": item}
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


class PubSub(Server):
    def __init__(
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        nodes: List[AbstractNode] = None,
        host: str = None,
        username: str = None,
        password: str = None,
        exchange_name: str = None,
    ) -> None:  # pylint: disable=too-many-arguments
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

        def callback(ch, method, properties, item):  # pylint: disable=unused-argument
            item = pickle.loads(item)
            self._process_nodes(item)
            return

        self.channel.basic_consume(
            queue=self.queue_name, on_message_callback=callback, auto_ack=True
        )
        self.channel.start_consuming()

        # clean up nodes with threads
        for node in self.pipeline.nodes:
            if node.name.endswith(".visual"):
                node.release_resources()


class Queue(Server):
    def __init__(
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        nodes: List[AbstractNode] = None,
        host: str = None,
        username: str = None,
        password: str = None,
        queue_name: str = None,
    ) -> None:  # pylint: disable=too-many-arguments
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

        def callback(ch, method, properties, item):  # pylint: disable=unused-argument
            item = pickle.loads(item)
            self._process_nodes(item)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=callback)
        self.channel.start_consuming()

        # clean up nodes with threads
        for node in self.pipeline.nodes:
            if node.name.endswith(".visual"):
                node.release_resources()


class ReqRes(Server):
    def __init__(
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        nodes: List[AbstractNode] = None,
        host: str = None,
        port: int = None,
    ) -> None:  # pylint: disable=too-many-arguments
        super().__init__(
            pipeline_path, config_updates_cli, custom_nodes_parent_subdir, nodes
        )
        self.app = FastAPI()
        self.host = host
        self.port = port

    def run(self) -> None:  # pylint: disable=too-many-branches
        """execute single or continuous inference"""

        # move outside?
        @self.app.post("/image")
        async def image(item: dict = Body):
            self._process_nodes(item)
            return

        # https://www.uvicorn.org/deployment/#running-programmatically
        # This doesn't work:
        # uvicorn.run("server:app", host="127.0.0.1", port=5000, log_level="info")
        # multiple workers with PKD depends on nodes used - e.g. media_writer needs to combine
        # correctly, weights downloading will have issues, tracking needs to be in sequence, etc.
        # If none of the above apply, async will be advantageous.
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

        # clean up nodes with threads
        for node in self.pipeline.nodes:
            if node.name.endswith(".visual"):
                node.release_resources()
