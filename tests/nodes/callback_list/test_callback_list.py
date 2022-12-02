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
from unittest import TestCase

import pytest

import peekingduck.nodes.dabble.fps as pkd_fps
import peekingduck.nodes.input.visual as pkd_visual
from peekingduck.nodes.callback_list import CallbackList
from peekingduck.runner import Runner
from tests.conftest import assert_msg_in_logs

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

                class CallbackClass:
                    def callback_method(self, data_pool):
                        raise ValueError("Method")

                    @classmethod
                    def callback_class_method(cls, data_pool):
                        raise ValueError("Class method")

                    @staticmethod
                    def callback_static_method(data_pool):
                        raise ValueError("Static method")

                callback_obj = CallbackClass()
                """
            )
        )
    yield


class CounterCallback:
    def __init__(self):
        self.count = 0

    def callback(self, data_pool):
        self.count += 1

    def halt_execution(self, data_pool):
        raise ValueError("Halt execution")


@pytest.mark.usefixtures("tmp_dir", "callback_file")
class TestCallbackList:
    def test_init_creates_empty_list_for_supported_events(self):
        callback_list = CallbackList()

        for event_type in SUPPORTED_EVENTS:
            assert event_type in callback_list.callbacks
            assert callback_list.callbacks[event_type] == []

        assert callback_list.EVENT_TYPES == SUPPORTED_EVENTS

    @pytest.mark.parametrize("event_type", SUPPORTED_EVENTS)
    def test_append_to_the_list_mapped_to_event_type(
        self, empty_callback_list, event_type
    ):
        callback = lambda data_pool: None

        empty_callback_list.append(event_type, callback)

        assert empty_callback_list.callbacks[event_type][-1] == callback
        for event in SUPPORTED_EVENTS:
            if event != event_type:
                assert empty_callback_list.callbacks[event] == []

    @pytest.mark.parametrize("event_type", SUPPORTED_EVENTS)
    @pytest.mark.parametrize(
        "callback_type,expected",
        [
            ("callback_func", "Function"),
            ("callback_obj::callback_method", "Method"),
            ("callback_obj::callback_class_method", "Class method"),
            ("callback_obj::callback_static_method", "Static method"),
            # Check that using ClassName works for class and static methods
            ("CallbackClass::callback_class_method", "Class method"),
            ("CallbackClass::callback_static_method", "Static method"),
        ],
    )
    def test_construct_from_dict(self, event_type, callback_type, expected):
        """Checks that creating CallbackList from dictionary works. Checks
        various ways to specify the callback methods.
        """
        callback_dict = {event_type: [f"my_callback::{callback_type}"]}
        callback_list = CallbackList.from_dict(callback_dict)
        on_event = getattr(callback_list, f"on_{event_type}")
        with pytest.raises(ValueError) as excinfo:
            on_event({})
        assert expected in str(excinfo.value)

    def test_skip_invalid_event_type(self):
        """Checks that invalid event key is skipped. Valid callbacks are still
        created.
        """
        callback_dict = {
            "run_begin": ["my_callback::callback_func"],
            "invalid_event": ["my_callback::callback_func"],
        }
        with pytest.raises(ValueError) as excinfo, TestCase.assertLogs(
            "peekingduck.pipeline.nodes.callback_list"
        ) as captured:
            callback_list = CallbackList.from_dict(callback_dict)
            callback_list.on_run_begin({})
        assert_msg_in_logs("event is not one of", captured.records)
        assert "Function" in str(excinfo.value)

    def test_skip_invalid_module(self):
        callback_dict = {
            "run_begin": [
                "my_callback::callback_func",
                "invalid_module::callback_func",
            ]
        }
        with pytest.raises(ValueError) as excinfo, TestCase.assertLogs(
            "peekingduck.pipeline.nodes.callback_list"
        ) as captured:
            callback_list = CallbackList.from_dict(callback_dict)
            callback_list.on_run_begin({})
        assert_msg_in_logs("is an invalid module", captured.records)
        assert "Function" in str(excinfo.value)

    @pytest.mark.parametrize(
        "callback_name",
        [
            "invalid_name",
            "callback_obj::invalid_name",
            "invalid_obj::callback_method",
            "CallbackClass::invalid_name",
        ],
    )
    def test_skip_invalid_function_name(self, callback_name):
        callback_dict = {
            "run_begin": ["my_callback::callback_func", f"my_callback::{callback_name}"]
        }
        with pytest.raises(ValueError) as excinfo, TestCase.assertLogs(
            "peekingduck.pipeline.nodes.callback_list"
        ) as captured:
            callback_list = CallbackList.from_dict(callback_dict)
            callback_list.on_run_begin({})
        assert_msg_in_logs("is not found", captured.records)
        assert "Function" in str(excinfo.value)

    @pytest.mark.parametrize("event_type", SUPPORTED_EVENTS)
    def test_run_event_is_triggered_once_per_node_run(
        self, create_input_video, event_type
    ):
        """Checks that callback events such as on_run_begin and on_run_end are
        triggered once per node per run iteration. CounterCallback tracks the
        number of times it is called internally.

        The count is equal to num_iter * num_nodes
        """
        num_iter = 5
        _ = create_input_video("video1.avi", fps=10, size=(600, 800, 3), num_frames=30)
        cb_obj = CounterCallback()
        visual_node = pkd_visual.Node(source="video1.avi")
        fps_node = pkd_fps.Node()
        visual_node.callback_list.append(event_type, cb_obj.callback)
        fps_node.callback_list.append(event_type, cb_obj.callback)
        runner = Runner(num_iter=num_iter, nodes=[visual_node, fps_node])

        runner.run()

        assert cb_obj.count == num_iter * 2

    def test_run_end_is_triggered_after_run_begin(self, create_input_video):
        """Checks that on_run_begin increments `count` by 1 before one_run_end
        terminates the run.
        """
        num_iter = 5
        _ = create_input_video("video1.avi", fps=10, size=(600, 800, 3), num_frames=30)
        cb_obj = CounterCallback()
        visual_node = pkd_visual.Node(source="video1.avi")
        fps_node = pkd_fps.Node()
        visual_node.callback_list.append("run_begin", cb_obj.callback)
        visual_node.callback_list.append("run_end", cb_obj.halt_execution)
        runner = Runner(num_iter=num_iter, nodes=[visual_node, fps_node])

        with pytest.raises(ValueError) as excinfo:
            runner.run()
        assert "Halt execution" in str(excinfo.value)
        assert cb_obj.count == 1
