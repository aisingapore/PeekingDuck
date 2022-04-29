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
CLI functions for PeekingDuck.
"""

import logging

import click

from peekingduck import __version__
from peekingduck.commands.core import init, run, verify_installation
from peekingduck.commands.create_node import create_node
from peekingduck.commands.nodes import nodes

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@click.group(invoke_without_command=True)
@click.version_option(__version__)
@click.option("--verify_install", is_flag=True, help="Verify PeekingDuck installation.")
@click.pass_context
def cli(ctx: click.Context, verify_install: bool) -> None:
    """PeekingDuck is a modular computer vision inference framework.

    Developed by Computer Vision Hub at AI Singapore.
    """
    if ctx.invoked_subcommand is None:
        if verify_install:
            verify_installation()
        else:
            print(ctx.get_help())


cli.add_command(create_node)
cli.add_command(init)
cli.add_command(nodes)
cli.add_command(run)
