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

import importlib

import pytest

from peekingduck.declarative_loader import PEEKINGDUCK_NODE_TYPES
from tests.conftest import PKD_DIR


@pytest.fixture(name="node_package_and_module", params=PEEKINGDUCK_NODE_TYPES)
def fixture_node_package_and_module(request):
    """Returns the node package which corresponds to one of ``PEEKINGDUCK_NODE_TYPES``
    and its corresponding node submodules.

    The node package yielded (for `dabble`) is equivalent to:
        from peekingduck.pipeline.nodes import dabble

    The corresponding node submodule names yielded is:
        ["bbox_count", "check_large_groups", ...]
    subpackages and "__init__" are omitted.
    """
    package = importlib.import_module(f".{request.param}", "peekingduck.pipeline.nodes")

    module_names = [
        path.stem
        for path in (PKD_DIR / "pipeline" / "nodes" / request.param).iterdir()
        if path.suffix == ".py" and not path.stem.startswith("_")
    ]
    yield package, module_names


def test_all_node_modules_are_surfaced(node_package_and_module):
    """Checks that importing the <node type> subpackage and subsequently
    instantiating node submodules works.

    Equivalent to:
        from peekingduck.pipeline.nodes import dabble

        _ = dabble.bbox_count.Node()

    Previously, it will fail with 'peekingduck.pipeline.nodes.dabble' has no
    'bbox_count' error.
    """
    package, module_names = node_package_and_module

    assert len(module_names) > 0
    for module_name in module_names:
        assert hasattr(package, module_name)
