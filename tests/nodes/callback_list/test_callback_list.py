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

import pytest

import peekingduck.pipeline.dabble.fps as pkd_fps
import peekingduck.pipeline.input.visual as pkd_visual
from peekingduck.pipeline.callback_list import CallbackList
from peekingduck.runner import Runner

SUPPORTED_EVENTS = ["run_begin", "run_end"]


@pytest.fixture(name="empty_callback_list")
def fixture_empty_callback_list():
    return CallbackList()


class CounterCallback:
    def __init__(self):
        self.count = 0

    def callback(self, data_pool):
        self.count += 1


@pytest.mark.usefixtures("tmp_dir")
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
