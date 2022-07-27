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

"""PeekingDuck CLI `create-node` command."""

import logging
from pathlib import Path

import click

from peekingduck.commands.create_node_helper import (
    create_config_and_script_files,
    ensure_relative_path,
    ensure_valid_name,
    ensure_valid_name_partial,
    ensure_valid_type,
    ensure_valid_type_partial,
    get_config_and_script_paths,
    verify_option,
)
from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPES, DeclarativeLoader

logger = logging.getLogger("peekingduck.cli")  # pylint: disable=invalid-name


@click.command()
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
    node_subdir: str = None,
    node_type: str = None,
    node_name: str = None,
    config_path: str = None,
) -> None:
    """Automates the creation of a new custom node.

    If the options `node_subdir`, `node_type`, or `node_name` are not
    specified, users will be prompted them for the values while performing
    checks to ensure value validity.
    """
    project_dir = Path.cwd()
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
                f"Enter node directory relative to {project_dir}/src",
                default="custom_nodes",
                value_proc=ensure_relative_path,
            )
        node_dir = project_dir / "src" / node_subdir

        node_type = verify_option(
            node_type, value_proc=ensure_valid_type_partial(node_type_choices)
        )
        if node_type is None:
            node_type = click.prompt(
                f"Select node type ({', '.join(node_type_choices.choices)})",
                value_proc=ensure_valid_type_partial(node_type_choices),
            )

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


def _create_nodes_from_config_file(
    config_path: str, project_dir: Path, node_type_choices: click.Choice
) -> None:
    """Creates custom nodes declared in the pipeline config file."""
    pipeline_path = Path(config_path)
    if not pipeline_path.is_absolute():
        pipeline_path = (project_dir / pipeline_path).resolve()
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Config file '{config_path}' is not found at {pipeline_path}!"
        )
    logger.info(f"Creating custom nodes declared in {pipeline_path}.")
    # Load run config with DeclarativeLoader to ensure consistency
    loader = DeclarativeLoader(pipeline_path, "None", "src")
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
            ensure_valid_type(node_type_choices, node_type)
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
