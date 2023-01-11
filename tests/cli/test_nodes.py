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

import io
import math
from pathlib import Path

import pytest
from click.testing import CliRunner

from peekingduck.cli import cli
from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPES

PKD_DIR = Path(__file__).resolve().parents[2] / "peekingduck"
PKD_CONFIG_DIR = PKD_DIR / "configs"


def available_nodes_msg(type_name=None, order="ASC"):
    def len_enumerate(item):
        return int(math.log10(item[0]) + 1) + len(item[1])

    output = io.StringIO()
    if type_name is None:
        node_types = PEEKINGDUCK_NODE_TYPES
    else:
        node_types = [type_name]

    url_prefix = "https://peekingduck.readthedocs.io/en/stable/nodes/"
    url_postfix = ".html#module-"
    for node_type in node_types:
        node_names = sorted(
            [path.stem for path in (PKD_CONFIG_DIR / node_type).glob("*.yml")],
            reverse=order == "DESC",
        )
        idx_and_node_names = list(enumerate(node_names, start=1))
        max_length = len_enumerate(max(idx_and_node_names, key=len_enumerate))
        print(f"\nPeekingDuck has the following {node_type} nodes:", file=output)
        for idx, node_name in idx_and_node_names:
            node_path = f"{node_type}.{node_name}"
            url = f"{url_prefix}{node_path}{url_postfix}{node_path}"
            node_width = max_length + 1 - int(math.log10(idx) + 1)
            print(f"{idx}:{node_name: <{node_width}}Info: {url}", file=output)
    print("\n", file=output)
    content = output.getvalue()
    output.close()
    return content


class TestCliNodes:
    @pytest.mark.parametrize("order", ["ASC", "DESC"])
    def test_nodes_all(self, order):
        result = CliRunner().invoke(cli, ["nodes", "--order", order])
        print(result.exception)
        print(result.output)
        assert result.exit_code == 0
        assert result.output == available_nodes_msg(order=order)

    @pytest.mark.parametrize("order", ["ASC", "DESC"])
    def test_nodes_single(self, order):
        for node_type in PEEKINGDUCK_NODE_TYPES:
            result = CliRunner().invoke(cli, ["nodes", node_type, "--order", order])
            assert result.exit_code == 0
            assert result.output == available_nodes_msg(node_type, order)
