import logging
from pathlib import Path

import click

from peekingduck.commands import LOGGER_NAME
from peekingduck.server import PubSub, Queue, ReqRes
from peekingduck.utils.deprecation import deprecate
from peekingduck.utils.logger import LoggerSetup

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name

config_path_option = click.option(
    "--config_path",
    default=None,
    type=click.Path(),
    help=(
        "List of nodes to run. None assumes pipeline_config.yml at current working directory"
    ),
)
log_level_option = click.option(
    "--log_level",
    default="info",
    help="""Modify log level {"critical", "error", "warning", "info", "debug"}""",
)
node_config_option = click.option(
    "--node_config",
    default="None",
    help="""Modify node configs by wrapping desired configs in a JSON string.\n
        Example: --node_config '{"node_name": {"param_1": var_1}}'""",
)
host_option = click.option("--host", default="0.0.0.0", help="""To be updated""")
username_option = click.option(
    "--username", default="peekingduck", help="""Username for RabbitMQ authentication"""
)
password_option = click.option(
    "--password",
    prompt=True,
    hide_input=True,
    help="""Password for RabbitMQ authentication""",
)


@click.command()
@config_path_option
@log_level_option
@node_config_option
@host_option
@username_option
@password_option
@click.option(
    "--exchange_name", default="pkd_exchange", help="""Name of RabbitMQ exchange"""
)
def pub_sub(  # pylint: disable=too-many-arguments
    config_path: str,
    log_level: str,
    node_config: str,
    host: str,
    username: str,
    password: str,
    exchange_name: str,
    nodes_parent_dir: str = "src",
) -> None:
    """Runs PeekingDuck Server in Publish-Subscribe Mode"""
    LoggerSetup.set_log_level(log_level)

    pipeline_config_path = _get_pipeline_config_path(config_path)
    logger.info("Launching PeekingDuck Server in Publish-Subscribe Mode")
    pkd_server = PubSub(
        pipeline_path=pipeline_config_path,
        config_updates_cli=node_config,
        custom_nodes_parent_subdir=nodes_parent_dir,
        host=host,
        username=username,
        password=password,
        exchange_name=exchange_name,
    )
    pkd_server.run()


@click.command()
@config_path_option
@log_level_option
@node_config_option
@host_option
@username_option
@password_option
@click.option("--queue_name", default="pkd_queue", help="""Name of RabbitMQ queue""")
def queue(  # pylint: disable=too-many-arguments
    config_path: str,
    log_level: str,
    node_config: str,
    host: str,
    username: str,
    password: str,
    queue_name: str,
    nodes_parent_dir: str = "src",
) -> None:
    """Runs PeekingDuck Server in Queue Mode"""
    LoggerSetup.set_log_level(log_level)

    pipeline_config_path = _get_pipeline_config_path(config_path)

    logger.info("Launching PeekingDuck Server in Queue Mode")
    pkd_server = Queue(
        pipeline_path=pipeline_config_path,
        config_updates_cli=node_config,
        custom_nodes_parent_subdir=nodes_parent_dir,
        host=host,
        username=username,
        password=password,
        queue_name=queue_name,
    )
    pkd_server.run()


@click.command()
@config_path_option
@log_level_option
@node_config_option
@host_option
@click.option("--port", default=5000, type=int, help="""Port to listen at""")
def req_res(  # pylint: disable=too-many-arguments
    config_path: str,
    log_level: str,
    node_config: str,
    host: str,
    port: int,
    nodes_parent_dir: str = "src",
) -> None:
    """Runs PeekingDuck Server in Request Response Mode"""
    LoggerSetup.set_log_level(log_level)

    pipeline_config_path = _get_pipeline_config_path(config_path)

    logger.info("Launching PeekingDuck Server in Request Response Mode")
    pkd_server = ReqRes(
        pipeline_path=pipeline_config_path,
        config_updates_cli=node_config,
        custom_nodes_parent_subdir=nodes_parent_dir,
        host=host,
        port=port,
    )
    pkd_server.run()


def _get_pipeline_config_path(config_path: str) -> Path:
    """
    TO-DO: this function can also be re-used for runner and viewer.
    """
    if config_path is None:
        curr_dir = Path.cwd()
        if (curr_dir / "pipeline_config.yml").is_file():
            config_path = curr_dir / "pipeline_config.yml"
        elif (curr_dir / "run_config.yml").is_file():
            deprecate(
                "using 'run_config.yml' as the default pipeline configuration "
                "file is deprecated and will be removed in the future. Please "
                "use 'pipeline_config.yml' instead.",
                2,
            )
            config_path = curr_dir / "run_config.yml"
        else:
            config_path = curr_dir / "pipeline_config.yml"
    return Path(config_path)
