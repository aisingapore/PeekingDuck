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

"""PeekingDuck CLI `nodes` command."""

import math
from pathlib import Path
from typing import Tuple

import click

from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPES


@click.command()
@click.argument("type_name", required=False)
def nodes(type_name: str = None) -> None:
    """Lists available nodes in PeekingDuck. When no argument is given, all
    available nodes will be listed. When the node type is given as an argument,
    all available nodes in the specified node type will be listed.

    Args:
        type_name (str): input, augment, model, draw, dabble, or output.
    """
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    if type_name is None:
        node_types = PEEKINGDUCK_NODE_TYPES
    else:
        node_types = [type_name]

    for node_type in node_types:
        click.secho("\nPeekingDuck has the following ", bold=True, nl=False)
        click.secho(f"{node_type} ", fg="red", bold=True, nl=False)
        click.secho("nodes:", bold=True)

        node_names = [path.stem for path in (config_dir / node_type).glob("*.yml")]
        idx_and_node_names = list(enumerate(node_names, start=1))

        max_length = _len_enumerate(max(idx_and_node_names, key=_len_enumerate))
        for idx, node_name in idx_and_node_names:
            url = _get_node_url(node_type, node_name)
            node_name_width = max_length + 1 - _num_digits(idx)

            click.secho(f"{idx}:", nl=False)
            click.secho(f"{node_name: <{node_name_width}}", bold=True, nl=False)
            click.secho("Info: ", fg="yellow", nl=False)
            click.secho(url)

    click.secho("\n")


def _get_node_url(node_type: str, node_name: str) -> str:
    """Constructs the URL to documentation of the specified node.

    Args:
        node_type (str): One of input, model, dabble, draw or output.
        node_name (str): Name of the node.

    Returns:
        (str): Full URL to the documentation of the specified node.
    """
    node_path = f"{node_type}.{node_name}"
    url_prefix = "https://peekingduck.readthedocs.io/en/stable/nodes/"
    url_postfix = ".html#module-"

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
    return _num_digits(item[0]) + len(item[1])


def _num_digits(number: int) -> int:
    """The number of digits of the given number.

    Args:
        number (int): The given number. It is assumed ``number > 0``.

    Returns:
        (int): Number of digits in the given number.
    """
    return int(math.log10(number) + 1)
