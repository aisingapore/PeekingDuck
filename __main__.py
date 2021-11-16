# Copyright 2021 AI Singapore

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
from peekingduck.utils.logger import LoggerSetup
import peekingduck.runner as pkd
import click


@click.command()
@click.option(
    "--config_path",
    default=None,
    type=click.Path(),
    help=(
        "List of nodes to run. None assumes run_config.yml at current working directory"
    ),
)
@click.option(
    "--log-level",
    default="info",
    help="""Modify log level {"error", "warning", "info", "debug"}""",
)
def run(config_path, log_level):
    if not config_path:
        pkd_dir = Path(__file__).parents[0]
        config_path = pkd_dir / "run_config.yml"

    LoggerSetup(log_level=log_level)
    logger = logging.getLogger(__name__)
    logger.info("Run path: %s", config_path)

    runner = pkd.Runner(config_path, "None", "PeekingDuck")
    runner.run()


if __name__ == "__main__":
    run()
