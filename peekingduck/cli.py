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
CLI functions for PeekingDuck.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import click
import yaml

from peekingduck import __version__
from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPES, DeclarativeLoader
from peekingduck.runner import Runner
from peekingduck.utils.create_node_helper import (
    create_config_and_script_files,
    ensure_relative_path,
    ensure_valid_name,
    ensure_valid_name_partial,
    get_config_and_script_paths,
    verify_option,
)
from peekingduck.utils.logger import LoggerSetup

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def create_custom_folder(custom_folder_name: str) -> None:
    """Makes custom nodes folder to create custom nodes.

    Args:
        custom_folder_name (:obj:`str`): Name of the custom nodes folder.
    """
    curr_dir = _get_cwd()
    custom_nodes_dir = curr_dir / "src" / custom_folder_name
    custom_nodes_config_dir = custom_nodes_dir / "configs"

    logger.info(f"Creating custom nodes folder in {custom_nodes_dir}")
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    custom_nodes_config_dir.mkdir(parents=True, exist_ok=True)


def create_yml() -> None:
    """Initialises the declarative *run_config.yml*."""
    # Default yml to be discussed
    default_yml = dict(nodes=["input.live", "model.yolo", "draw.bbox", "output.screen"])

    with open("run_config.yml", "w") as yml_file:
        yaml.dump(default_yml, yml_file, default_flow_style=False)


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """
    PeekingDuck is a modular computer vision inference framework.

    Developed by Computer Vision Hub at AI Singapore.
    """


@cli.command()
@click.option("--custom_folder_name", default="custom_nodes")
def init(custom_folder_name: str) -> None:
    """Initialise a PeekingDuck project"""
    print("Welcome to PeekingDuck!")
    create_custom_folder(custom_folder_name)
    create_yml()


@cli.command()
@click.option(
    "--config_path",
    default=None,
    type=click.Path(),
    help=(
        "List of nodes to run. None assumes run_config.yml at current working directory"
    ),
)
@click.option(
    "--node_config",
    default="None",
    help="""Modify node configs by wrapping desired configs in a JSON string.\n
        Example: --node_config '{"node_name": {"param_1": var_1}}'""",
)
@click.option(
    "--log_level",
    default="info",
    help="""Modify log level {"critical", "error", "warning", "info", "debug"}""",
)
def run(
    config_path: str, node_config: str, log_level: str, nodes_parent_dir: str = "src"
) -> None:
    """Runs PeekingDuck"""
    LoggerSetup.set_log_level(log_level)

    curr_dir = _get_cwd()
    if config_path is None:
        run_config_path = curr_dir / "run_config.yml"
    else:
        run_config_path = Path(config_path)

    runner = Runner(run_config_path, node_config, nodes_parent_dir)
    runner.run()


@cli.command()
@click.option(
    "--node_subdir",
    help=(
        "Path to the custom nodes directory, relative to the directory where "
        "the command is invoked."
    ),
    required=False,
)
@click.option(
    "--node_type",
    help=(
        "Node type, only accepts values from existing node types defined in "
        f"{PEEKINGDUCK_NODE_TYPES}."
    ),
    required=False,
)
@click.option(
    "--node_name",
    help=(
        "\b\nName of new custom node. The name cannot be a duplicate of an \n"
        "existing custom node. The name has the following requirements:\n"
        "- Minimum 2 characters\n"
        "- Can only contain alphanumeric characters, dashes and underscores \n"
        "  /[[a-zA-Z0-9_\\-]/\n"
        "- Must start with an alphabet\n"
        "- Must end with an alphanumeric character"
    ),
    required=False,
)
@click.option(
    "--config_path",
    help=(
        "List of nodes to parse. Will automatically create the config and script file "
        "for any custom nodes."
    ),
    required=False,
)
def create_node(
    node_subdir: Optional[str] = None,
    node_type: Optional[str] = None,
    node_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> None:
    """Automates the creation of a new custom node.

    If the options `node_subdir`, `node_type`, or `node_name` are not
    specified, users will be prompted them for the values while performing
    checks to ensure value validity.
    """
    project_dir = _get_cwd()
    node_type_choices = click.Choice(PEEKINGDUCK_NODE_TYPES)

    if config_path is not None and any(
        arg is not None for arg in (node_subdir, node_type, node_name)
    ):
        raise ValueError(
            "--config_path cannot be use with --node_subdir, --node_type, or "
            "--node_name!"
        )
    if config_path is not None:
        _create_nodes_from_config_file(config_path, project_dir, node_type_choices)
    else:
        click.secho("Creating new custom node...")
        node_subdir = verify_option(node_subdir, value_proc=ensure_relative_path)
        if node_subdir is None:
            node_subdir = click.prompt(
                f"Enter node directory relative to {project_dir}",
                default="src/custom_nodes",
                value_proc=ensure_relative_path,
            )
        node_dir = project_dir / node_subdir

        node_type = verify_option(
            node_type, value_proc=click.types.convert_type(node_type_choices)
        )
        if node_type is None:
            node_type = click.prompt("Select node type", type=node_type_choices)

        node_name = verify_option(
            node_name, value_proc=ensure_valid_name_partial(node_dir, node_type)
        )
        if node_name is None:
            node_name = click.prompt(
                "Enter node name",
                default="my_custom_node",
                value_proc=ensure_valid_name_partial(node_dir, node_type),
            )

        created_paths = get_config_and_script_paths(
            node_dir, ("configs", node_type), node_type, node_name
        )
        click.echo(f"\nNode directory:\t{node_dir}")
        click.echo(f"Node type:\t{node_type}")
        click.echo(f"Node name:\t{node_name}")
        click.echo("\nCreating the following files:")
        click.echo(f"\tConfig file: {created_paths['config']}")
        click.echo(f"\tScript file: {created_paths['script']}")

        proceed = click.confirm("Proceed?", default=True)
        if proceed:
            create_config_and_script_files(created_paths)
            click.echo("Created node!")
        else:
            click.echo("Aborted!")


@cli.command()
@click.argument("type_name", required=False)
def nodes(type_name: str = None) -> None:
    """Lists available nodes in PeekingDuck. When no argument is given, all
    available nodes will be listed. When the node type is given as an argument,
    all available nodes in the specified node type will be listed.

    Args:
        type_name (str): input, model, dabble, draw or output
    """
    config_dir = Path(__file__).resolve().parent / "configs"

    if type_name is None:
        node_types = ["input", "model", "dabble", "draw", "output"]
    else:
        node_types = [type_name]

    for node_type in node_types:
        click.secho("\nPeekingDuck has the following ", bold=True, nl=False)
        click.secho(f"{node_type} ", fg="red", bold=True, nl=False)
        click.secho("nodes:", bold=True)

        node_names = [path.stem for path in (config_dir / node_type).glob("*.yml")]
        max_length = _len_enumerate(max(enumerate(node_names), key=_len_enumerate))
        for num, node_name in enumerate(node_names):
            idx = num + 1
            url = _get_node_url(node_type, node_name)
            node_name_width = max_length + 1 - _num_digits(idx)

            click.secho(f"{idx}:", nl=False)
            click.secho(f"{node_name: <{node_name_width}}", bold=True, nl=False)
            click.secho("Info: ", fg="yellow", nl=False)
            click.secho(url)

    click.secho("\n")


def _create_nodes_from_config_file(
    config_path: str, project_dir: Path, node_type_choices: click.Choice
) -> None:
    """Creates custom nodes declared in the config file."""
    run_config_path = Path(config_path)
    if not run_config_path.is_absolute():
        run_config_path = (project_dir / run_config_path).resolve()
    if not run_config_path.exists():
        raise FileNotFoundError(
            f"Config file '{config_path}' is not found at {run_config_path}!"
        )
    logger.info(f"Creating custom nodes declared in {run_config_path}.")
    # Load run config with DeclarativeLoader to ensure consistency
    loader = DeclarativeLoader(run_config_path, "None", "src")
    try:
        node_subdir = project_dir / "src" / loader.custom_nodes_dir
    except AttributeError as custom_nodes_no_exist:
        raise ValueError(
            f"Config file '{config_path}' does not contain custom nodes!"
        ) from custom_nodes_no_exist

    for node_str, _ in loader.node_list:
        try:
            node_subdir, node_type, node_name = node_str.split(".")
        except ValueError:
            continue
        try:
            # Check node string formatting
            ensure_relative_path(node_subdir)
            node_dir = project_dir / "src" / node_subdir
            click.types.convert_type(node_type_choices)(node_type)
            ensure_valid_name(node_dir, node_type, node_name)
        except click.exceptions.UsageError as err:
            logger.warning(
                f"{node_str} contains invalid formatting: '{err.message}'. "
                "Skipping..."
            )
            continue
        created_paths = get_config_and_script_paths(
            node_dir, ("configs", node_type), node_type, node_name
        )
        logger.info(
            f"Creating files for {node_str}:\n\t"
            f"Config file: {created_paths['config']}\n\t"
            f"Script file: {created_paths['script']}"
        )
        create_config_and_script_files(created_paths)


def _get_cwd() -> Path:
    return Path.cwd()


def _get_node_url(node_type: str, node_name: str) -> str:
    """Constructs the URL to documentation of the specified node.

    Args:
        node_type (str): One of input, model, dabble, draw or output.
        node_name (str): Name of the node.

    Returns:
        (str): Full URL to the documentation of the specified node.
    """
    node_path = f"peekingduck.pipeline.nodes.{node_type}.{node_name}.Node"
    url_prefix = "https://peekingduck.readthedocs.io/en/stable/"
    url_postfix = ".html#"

    return f"{url_prefix}{node_path}{url_postfix}{node_path}"


def _len_enumerate(item: Tuple) -> int:
    """Calculates the string length of an enumerate item while accounting for
    the number of digits of the index.

    Args:
        item (Tuple): An item return by ``enumerate``, contains
            ``(index, element)`` where ``element`` is a ``str``.

    Returns:
        (int): Sum of length of ``element`` and the number of digits of its
            index.
    """
    return _num_digits(item[0] + 1) + len(item[1])


def _num_digits(number: int) -> int:
    """The number of digits of the given number.

    Args:
        number (int): The given number. It is assumed ``number > 0``.

    Returns:
        (int): Number of digits in the given number.
    """
    return int(math.log10(number))
