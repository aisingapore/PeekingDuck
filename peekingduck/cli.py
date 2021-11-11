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
CLI functions for Peekingduck
"""

import functools
import logging
import math
import re
from pathlib import Path
from typing import Callable, Optional, Tuple

import click
import yaml

from peekingduck import __version__
from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPE
from peekingduck.runner import Runner
from peekingduck.utils.logger import LoggerSetup

LoggerSetup()
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """
    PeekingDuck is a modular computer vision inference framework.

    Developed by Computer Vision Hub at AI Singapore.
    """


def _get_cwd() -> Path:
    return Path.cwd()


def _get_node_url(node_type: str, node_name: str) -> str:
    """Constructs the URL to documentation of the specified node

    Args:
        node_type (`str`): One of input, model, dabble, draw or output
        node_name (`str`): Name of the node

    Returns:
        (`str`): Full URL to the documentation of the specified node
    """
    node_path = f"peekingduck.pipeline.nodes.{node_type}.{node_name}.Node"
    url_prefix = "https://peekingduck.readthedocs.io/en/stable/"
    url_postfix = ".html#"

    return f"{url_prefix}{node_path}{url_postfix}{node_path}"


def _len_enumerate(item: Tuple) -> int:
    """Calculate the string length of an enumerate item while accounting for
    the number of digits of the index

    Args:
        item (`Tuple`): An item return by `enumerate`, contains
            `(index, element)` where `element` is a `str`.

    Returns:
        (`int`): Sum of length of `element` and the number of digits of its
            index
    """
    return _num_digits(item[0] + 1) + len(item[1])


def _num_digits(number: int) -> int:
    """Return the number of digits of the given number

    Args:
        number (`int`): The given number. It is assumed `number > 0`.

    Returns:
        (`int`): Number of digits in the given number
    """
    return int(math.log10(number))


def create_custom_folder(custom_folder_name: str) -> None:
    """Make custom nodes folder to create custom nodes"""
    curdir = _get_cwd()
    custom_folder_dir = curdir / "src" / custom_folder_name
    custom_configs_dir = custom_folder_dir / "configs"

    logger.info("Creating custom nodes folder in %s", custom_folder_dir)
    custom_folder_dir.mkdir(parents=True, exist_ok=True)
    custom_configs_dir.mkdir(parents=True, exist_ok=True)


def create_yml() -> None:
    """Inits the declarative yaml"""
    # Default yml to be discussed
    default_yml = dict(nodes=["input.live", "model.yolo", "draw.bbox", "output.screen"])

    with open("run_config.yml", "w") as yml_file:
        yaml.dump(default_yml, yml_file, default_flow_style=False)


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
def run(config_path: str, node_config: str) -> None:
    """Runs PeekingDuck"""

    curdir = _get_cwd()
    if config_path is None:
        run_path = curdir / "run_config.yml"
    else:
        run_path = Path(config_path)

    runner = Runner(run_path, node_config, "src")
    runner.run()


def _verify_input(value: Optional[str], value_proc: Callable) -> Optional[str]:
    """Verifies that input value via click.option matches the expected value.

    This sets ``value`` to ``None`` if it is invalid so the rest of the prompt
    can flow smoothly.

    Args:
        value (Optional[str]): Input value.
        value_proc (Callable): A function to check the validity of ``value``.

    Returns:
        (Optional[str]): ``value`` if it is a valid value. ``None`` if it is
            not.

    Raises:
        click.exceptions.UsageError: When ``value`` is invalid.
    """
    if value is not None:
        try:
            value = value_proc(value)
        except click.exceptions.UsageError as error:
            click.echo(f"Error: {error.message}", err=True)
            value = None
    return value


def _ensure_relative_path(node_subdir: str) -> str:
    """Checks that the subdir path does not contain parent directory
    navigator (..), is not absolute, and is not "peekingduck/pipeline/nodes".
    """
    pkd_node_subdir = "peekingduck/pipeline/nodes"
    if ".." in node_subdir:
        raise click.exceptions.UsageError("Path cannot contain '..'!")
    if Path(node_subdir).is_absolute():
        raise click.exceptions.UsageError("Path cannot be absolute!")
    if node_subdir == pkd_node_subdir:
        raise click.exceptions.UsageError(f"Path cannot be '{pkd_node_subdir}'!")
    return node_subdir


def _ensure_valid_name(node_dir: Path, node_type: str, node_name: str) -> str:
    if re.match(r"^[a-zA-Z][\w\-]*[^\W_]$", node_name) is None:
        raise click.exceptions.UsageError("Invalid node name!")
    if (node_dir / node_type / f"{node_name}.py").exists():
        raise click.exceptions.UsageError("Node name already exists!")
    return node_name


@cli.command()
@click.option("--node_subdir", required=False)
@click.option("--node_type", required=False)
@click.option("--node_name", required=False)
def create_node(
    node_subdir: Optional[str] = None,
    node_type: Optional[str] = None,
    node_name: Optional[str] = None,
) -> None:
    """Automates the creation of a new custom node.

    If the user does not specifiy ``node_subdir``, ``node_type``, or
    ``node_name`` through CLI, prompt them for the values while performing
    checks to ensure value validity.

    Args:
        node_subdir (Optional[str]): Path to the custom nodes directory,
            relative to the directory where the command is invoked.
        node_type (Optional[str]): Node type, only accepts values from existing
            node types defined in ``PEEKINGDUCK_NODE_TYPE``.
        node_name (Optional[str]): Name of new custom node. The name cannot be
            a duplicate of an existing custom node. The name has the following
            requirements:
                - Minimum 2 characters
                - Can only contain alphanumeric characters, dashes and
                    underscores /[[a-zA-Z0-9_\\-]/
                - Must start with a alphabet
                - Must end with an alphanumeric character
    """
    click.secho("Creating new custom node...")
    project_dir = Path.cwd()
    node_type_choices = click.Choice(PEEKINGDUCK_NODE_TYPE)

    node_subdir = _verify_input(node_subdir, value_proc=_ensure_relative_path)
    if node_subdir is None:
        node_subdir = click.prompt(
            f"Enter node directory relative to {project_dir}",
            default="src/custom_nodes",
            value_proc=_ensure_relative_path,
        )
    node_dir = project_dir / node_subdir

    node_type = _verify_input(
        node_type, value_proc=click.types.convert_type(node_type_choices)
    )
    if node_type is None:
        node_type = click.prompt("Select node type", type=node_type_choices)

    partial_ensure_valid_name = functools.partial(
        _ensure_valid_name, node_dir, node_type
    )
    if node_name is None:
        node_name = click.prompt(
            "Enter node name", value_proc=partial_ensure_valid_name
        )

    config_path = node_dir / "configs" / node_type / f"{node_name}.yml"
    script_path = node_dir / node_type / f"{node_name}.py"
    click.echo(f"\nNode directory:\t{node_dir}")
    click.echo(f"Node type:\t{node_type}")
    click.echo(f"Node name:\t{node_name}")
    click.echo("\nCreating the following files:")
    click.echo(f"\tConfig file: {config_path}")
    click.echo(f"\tScript file: {script_path}")

    proceed = click.confirm("Proceed?", default=True)
    if proceed:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        pkd_dir = Path(__file__).resolve().parent
        template_name = "node_template"
        with open(pkd_dir / "configs" / f"{template_name}.yml") as template_file, open(
            config_path, "w"
        ) as outfile:
            outfile.write(template_file.read())
        with open(
            pkd_dir / "pipeline" / "nodes" / f"{template_name}.py"
        ) as template_file, open(script_path, "w") as outfile:
            outfile.write(template_file.read())
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
    configs_dir = Path(__file__).resolve().parent / "configs"

    if type_name is None:
        node_types = ["input", "model", "dabble", "draw", "output"]
    else:
        node_types = [type_name]

    for node_type in node_types:
        click.secho("\nPeekingDuck has the following ", bold=True, nl=False)
        click.secho(f"{node_type} ", fg="red", bold=True, nl=False)
        click.secho("nodes:", bold=True)

        node_names = [path.stem for path in (configs_dir / node_type).glob("*.yml")]
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
