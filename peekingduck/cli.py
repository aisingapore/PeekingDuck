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

import functools
import logging
import math
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import click
import yaml

from peekingduck import __version__
from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPES
from peekingduck.runner import Runner
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
def run(config_path: str, node_config: str, log_level: str) -> None:
    """Runs PeekingDuck"""
    LoggerSetup.set_log_level(log_level)

    curr_dir = _get_cwd()
    if config_path is None:
        run_config_path = curr_dir / "run_config.yml"
    else:
        run_config_path = Path(config_path)

    runner = Runner(run_config_path, node_config, "src")
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
def create_node(
    node_subdir: Optional[str] = None,
    node_type: Optional[str] = None,
    node_name: Optional[str] = None,
) -> None:
    """Automates the creation of a new custom node.

    If the options `node_subdir`, `node_type`, or `node_name` are not
    specified, users will be prompted them for the values while performing
    checks to ensure value validity.
    """
    click.secho("Creating new custom node...")
    project_dir = Path.cwd()
    node_type_choices = click.Choice(PEEKINGDUCK_NODE_TYPES)

    node_subdir = _verify_option(node_subdir, value_proc=_ensure_relative_path)
    if node_subdir is None:
        node_subdir = click.prompt(
            f"Enter node directory relative to {project_dir}",
            default="src/custom_nodes",
            value_proc=_ensure_relative_path,
        )
    node_dir = project_dir / node_subdir

    node_type = _verify_option(
        node_type, value_proc=click.types.convert_type(node_type_choices)
    )
    if node_type is None:
        node_type = click.prompt("Select node type", type=node_type_choices)

    node_name = _verify_option(
        node_name, value_proc=_ensure_valid_name_partial(node_dir, node_type)
    )
    if node_name is None:
        node_name = click.prompt(
            "Enter node name",
            default="my_custom_node",
            value_proc=_ensure_valid_name_partial(node_dir, node_type),
        )

    created_paths = _get_config_and_script_paths(
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
        created_paths["config"].parent.mkdir(parents=True, exist_ok=True)
        created_paths["script"].parent.mkdir(parents=True, exist_ok=True)
        template_paths = _get_config_and_script_paths(
            Path(__file__).resolve().parent,
            "configs",
            ("pipeline", "nodes"),
            "node_template",
        )
        with open(template_paths["config"]) as template_file, open(
            created_paths["config"], "w"
        ) as outfile:
            outfile.write(template_file.read())
        with open(template_paths["script"]) as template_file, open(
            created_paths["script"], "w"
        ) as outfile:
            lines = template_file.readlines()
            start = -1
            for i, line in enumerate(lines):
                if line.startswith('"'):
                    start = i
                    break
            # In case we couldn't find a starting line
            start = max(0, start)
            outfile.writelines(lines[start:])
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


def _ensure_valid_name_partial(node_dir: Path, node_type: str) -> Callable:
    return functools.partial(_ensure_valid_name, node_dir, node_type)


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


def _get_config_and_script_paths(
    parent_dir: Path,
    config_subdir: Union[str, Tuple[str, ...]],
    script_subdir: Union[str, Tuple[str, ...]],
    file_stem: str,
) -> Dict[str, Path]:
    """Returns the node config file and its corresponding script file."""
    if isinstance(config_subdir, tuple):
        config_subpath = Path(*config_subdir)
    else:
        config_subpath = Path(config_subdir)
    if isinstance(script_subdir, tuple):
        script_subpath = Path(*script_subdir)
    else:
        script_subpath = Path(script_subdir)

    return {
        "config": parent_dir / config_subpath / f"{file_stem}.yml",
        "script": parent_dir / script_subpath / f"{file_stem}.py",
    }


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


def _verify_option(value: Optional[str], value_proc: Callable) -> Optional[str]:
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
    if value is None:
        return value
    try:
        value = value_proc(value)
    except click.exceptions.UsageError as error:
        click.echo(f"Error: {error.message}", err=True)
        value = None
    return value
