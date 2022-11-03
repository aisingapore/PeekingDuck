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

"""
Container for callbacks which are called at certain points in the
pipeline.
"""

import importlib
from typing import Any, Callable, Dict, List


class CallbackList:
    """Container for callback functions."""

    def __init__(self) -> None:
        self.callbacks: Dict[str, List[Callable]] = {
            "run_begin": [],
            "run_end": [],
        }

    def append(self, event_type: str, callback: Callable) -> None:
        """Adds a callback to the specified `event_type`.

        Args:
            event_type (str): Determines the point in the run loop at which the
                callback is called.
            callback (Callable): A function which can take in a dictionary as
                an argument.
        """
        self.callbacks[event_type].append(callback)

    def on_run_begin(self, pipeline_data: Dict[str, Any]) -> None:
        """Triggers all callbacks set to run at the `run_begin` event.

        Args:
            pipeline_data (Dict[str, Any]): The current pipeline data.
        """
        self._on_event("run_begin", pipeline_data)

    def on_run_end(self, pipeline_data: Dict[str, Any]) -> None:
        """Triggers all callbacks set to run at the `run_end` event.

        Args:
            pipeline_data (Dict[str, Any]): The current pipeline data.
        """
        self._on_event("run_end", pipeline_data)

    def _on_event(self, event_type: str, pipeline_data: Dict[str, Any]) -> None:
        """Triggers all callbacks based on the specified event.

        Args:
            event_type (str): The specified event.
            pipeline_data (Dict[str, Any]): The current pipeline data.
        """
        for callback in self.callbacks[event_type]:
            callback(pipeline_data)

    @classmethod
    def from_dict(cls, callback_dict: Dict[str, List[str]]) -> "CallbackList":
        """Constructs a `CallbackList` object populated with callbacks from
        `callback_dict`.

        Args:
            callback_dict (Dict[str, List[str]]): A dictionary defined in the
                pipeline config file. Maps `event_type` to a list of callback
                definitions. Each callback defintion should contain:
                    <module name>[.<submodule name>]*.<callback function name>
                The callback modules are expected to be found in the
                "callbacks" directory in the same location as the pipeline
                config file.
        """
        callback_list = cls()
        base_module = "callbacks"
        for event_type, callback_names in callback_dict.items():
            for callback_name in callback_names:
                module_part, *callback_parts = callback_name.split("::")
                module = importlib.import_module(f"{base_module}.{module_part}")
                callback = lambda *args: None
                for part in callback_parts:
                    callback = getattr(module, part)
                callback_list.append(event_type, callback)

        return callback_list
