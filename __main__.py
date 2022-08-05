# Copyright 2022 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Workaround for running Peekingduck from project directory
"""

import logging
from pathlib import Path

import click

from peekingduck.cli import cli, run, serve


# @cli.command()
# @click.option(
#     "--config_path",
#     default=None,
#     type=click.Path(),
#     help=(
#         "List of nodes to run. None assumes pipeline_config.yml is in the same "
#         "directory as __main__.py"
#     ),
# )
# @click.option(
#     "--log_level",
#     default="info",
#     help="""Modify log level {"critical", "error", "warning", "info", "debug"}""",
# )
# @click.option(
#     "--num_iter",
#     default=None,
#     type=int,
#     help="Stop pipeline after running this number of iterations",
# )
# @click.option(
#     "--viewer",
#     default=False,
#     is_flag=True,
#     help="Launch PeekingDuck viewer",
# )
# @click.pass_context
# def main(
#     context: click.Context,
#     config_path: str,
#     log_level: str,
#     num_iter: int,
#     viewer: bool,
# ) -> None:
#     """Invokes the run() CLI command with some different defaults for
#     ``node_config`` and ``nodes_parent_dir``.
#     """
#     if config_path is None:
#         pkd_dir = Path(__file__).resolve().parent
#         if (pkd_dir / "pipeline_config.yml").is_file():
#             config_path = str(pkd_dir / "pipeline_config.yml")
#         elif (pkd_dir / "run_config.yml").is_file():
#             config_path = str(pkd_dir / "run_config.yml")
#         else:
#             config_path = str(pkd_dir / "pipeline_config.yml")
#         nodes_parent_dir = pkd_dir.name
#     else:
#         nodes_parent_dir = "src"

#     logger = logging.getLogger(__name__)
#     logger.info(f"Run path: {config_path}")

#     context.invoke(
#         run,
#         config_path=config_path,
#         node_config="None",
#         log_level=log_level,
#         num_iter=num_iter,
#         nodes_parent_dir=nodes_parent_dir,
#         viewer=viewer,
#     )

# Commented out above and using this as a quick way to test PKD server functionality without having
# to pip install . every time.
@cli.command()
@click.option(
    "--config_path",
    default=None,
    type=click.Path(),
    help=(
        "List of nodes to run. None assumes pipeline_config.yml is in the same "
        "directory as __main__.py"
    ),
)
@click.option(
    "--log_level",
    default="info",
    help="""Modify log level {"critical", "error", "warning", "info", "debug"}""",
)
@click.option(
    "--mode",
    default="request-response",
    help="""Choose server mode {"request-response", "message-queue", "publish-subscribe"}""",
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="""Host IP address to listen at""",
)
@click.option(
    "--port",
    default=5000,
    type=int,
    help="""Port to listen to""",
)
@click.pass_context
def main(
    context: click.Context,
    config_path: str,
    log_level: str,
    mode: str,
    host: str,
    port: int,
) -> None:
    """Invokes the run() CLI command with some different defaults for
    ``node_config`` and ``nodes_parent_dir``.
    """
    if config_path is None:
        pkd_dir = Path(__file__).resolve().parent
        if (pkd_dir / "pipeline_config.yml").is_file():
            config_path = str(pkd_dir / "pipeline_config.yml")
        elif (pkd_dir / "run_config.yml").is_file():
            config_path = str(pkd_dir / "run_config.yml")
        else:
            config_path = str(pkd_dir / "pipeline_config.yml")
        nodes_parent_dir = pkd_dir.name
    else:
        nodes_parent_dir = "src"

    logger = logging.getLogger(__name__)
    logger.info(f"Run path: {config_path}")

    context.invoke(
        serve,
        config_path=config_path,
        node_config="None",
        log_level=log_level,
        mode=mode,
        host=host,
        port=port,
        nodes_parent_dir=nodes_parent_dir,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
