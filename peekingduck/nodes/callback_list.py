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

import functools
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List


class CallbackList:
    """Container for callback functions."""

    BASE_MODULE = "callbacks"
    EVENT_TYPES = ["run_begin", "run_end"]
    logger = logging.getLogger(__name__)

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
                definitions. Each callback definition should contain:
                    <module>[.<submodule>]*[::<callback class>]::<callback function>
                The callback modules are expected to be found in the
                "callbacks" directory in the same location as the pipeline
                config file.
        """
        callback_dir = Path.cwd() / cls.BASE_MODULE
        callback_list = cls()
        for event_type, callback_names in callback_dict.items():
            for callback_name in callback_names:
                module_part, *callback_parts = callback_name.split("::")
                try:
                    module = import_module_from_file_location(module_part, callback_dir)
                    callback = chain_getattr(module, callback_parts)
                    callback_list.append(event_type, callback)
                except KeyError:
                    cls.logger.warning(
                        f"Skipping `{{'{event_type}': {callback_names}}}` as "
                        f"`{event_type}` event is not one of {cls.EVENT_TYPES}."
                    )
                except FileNotFoundError:
                    cls.logger.warning(
                        f"Skipping `{callback_name}` as `{module_part}` is an invalid module."
                    )
                except AttributeError:
                    cls.logger.warning(
                        f"Skipping `{callback_name}` as `{'::'.join(callback_parts)}` is not found."
                    )
        return callback_list


def chain_getattr(obj: object, names: List[str], *args: Any) -> Any:
    """Chains getattr according to number of elements in names.

    Args:
        obj (object): The object to successively get attributes from.
        names (List[str]): The attribute names.

    Returns:
        (Any): The final desired attribute.
    """

    def _getattr(obj: object, name: str) -> Any:
        return getattr(obj, name, *args)

    return functools.reduce(_getattr, names, obj)


def import_module_from_file_location(module_name: str, location: Path) -> ModuleType:
    """Imports module from the specified location without having to modify
    `sys.path`.

    Args:
        module_name (str): Name of module, should be the same as the name of
            the Python script file and its subdirectory relative to `callbacks`.
        location (Path): The directory where the Python script file containing
            the module can be found.

    Returns:
        (ModuleType): The imported module.

    Raises:
        AssertionError: If the default file loader from importlib.util cannot
        be used.
    """
    spec = importlib.util.spec_from_file_location(
        module_name, location.joinpath(*module_name.split(".")).with_suffix(".py")
    )
    # use default loader
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
