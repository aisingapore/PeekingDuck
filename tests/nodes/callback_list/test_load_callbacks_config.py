# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import textwrap
from pathlib import Path

import pytest

import peekingduck.nodes.input.visual as pkd_visual
from peekingduck.nodes.callback_list import CallbackList
from peekingduck.runner import Runner

SUPPORTED_EVENTS = ["run_begin", "run_end"]


@pytest.fixture(name="empty_callback_list")
def fixture_empty_callback_list():
    return CallbackList()


@pytest.fixture(name="callback_file")
def fixture_callback_file():
    callback_dir = Path.cwd() / "callbacks"
    callback_dir.mkdir()
    with open(callback_dir / "my_callback.py", "w") as outfile:
        outfile.write(
            textwrap.dedent(
                """
                def callback_func(data_pool):
                    raise ValueError("Function")
                """
            )
        )
    yield


@pytest.fixture(name="pipeline_file")
def fixture_pipeline_file():
    pipeline_path = Path.cwd() / "pipeline_config.yml"
    with open(pipeline_path, "w") as outfile:
        outfile.write(
            textwrap.dedent(
                """
                nodes:
                - input.visual:
                    source: video1.avi
                    callbacks:
                      run_begin: [my_callback::callback_func]
                """
            )
        )
    yield pipeline_path


@pytest.fixture(name="default_pipeline_file")
def fixture_default_pipeline_file():
    pipeline_path = Path.cwd() / "pipeline_config.yml"
    with open(pipeline_path, "w") as outfile:
        outfile.write(
            textwrap.dedent(
                """
                nodes:
                - input.visual:
                    source: video1.avi
                """
            )
        )
    yield pipeline_path


@pytest.mark.usefixtures("tmp_dir", "callback_file")
class TestLoadCallbacksConfig:
    def test_declarative_loader_parses_callbacks_in_pipeline_config(
        self, create_input_video, pipeline_file
    ):
        _ = create_input_video("video1.avi", fps=10, size=(600, 800, 3), num_frames=30)
        runner = Runner(
            pipeline_path=pipeline_file,
            config_updates_cli="None",
            custom_nodes_parent_subdir="src",
        )
        with pytest.raises(ValueError) as excinfo:
            runner.run()
        assert "Function" in str(excinfo.value)

    def test_declarative_loader_parses_callbacks_in_cli_config(
        self, create_input_video, default_pipeline_file
    ):
        _ = create_input_video("video1.avi", fps=10, size=(600, 800, 3), num_frames=30)
        runner = Runner(
            pipeline_path=default_pipeline_file,
            config_updates_cli=(
                "{'input.visual': {'callbacks': "
                "{'run_begin': ['my_callback::callback_func']}}}"
            ),
            custom_nodes_parent_subdir="src",
        )
        with pytest.raises(ValueError) as excinfo:
            runner.run()
        assert "Function" in str(excinfo.value)

    def test_declarative_loader_parses_callbacks_config_in_node_constructor(
        self, create_input_video
    ):
        _ = create_input_video("video1.avi", fps=10, size=(600, 800, 3), num_frames=30)
        visual_node = pkd_visual.Node(
            source="video1.avi", callbacks={"run_begin": ["my_callback::callback_func"]}
        )
        runner = Runner(nodes=[visual_node])

        with pytest.raises(ValueError) as excinfo:
            runner.run()
        assert "Function" in str(excinfo.value)
