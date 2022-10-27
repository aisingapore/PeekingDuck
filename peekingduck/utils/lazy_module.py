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

"""Module class for lazy import to defer optional requirement checking."""

import importlib
from importlib.machinery import ModuleSpec
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

ImportStructure = Dict[str, List[str]]


class _LazyModule(ModuleType):
    """Module class that surfaces all objects but only performs associated
    imports when the objects are requested.

    Adapted from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py#L1023
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        module_file: str,
        import_structure: ImportStructure,
        module_spec: Optional[ModuleSpec] = None,
        extra_objects: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name)

        self.__file__: str = module_file
        self.__path__ = [str(Path(module_file).resolve().parent)]
        self.__spec__ = module_spec
        self._import_structure = import_structure
        self._objects = {} if extra_objects is None else extra_objects

        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(
            chain(*import_structure.values())
        )

    def __dir__(self) -> Iterable[str]:
        """Returns a list of attributes with lazy modules and classes appended.
        Needed for autocompletion in an IDE.
        """
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)  # type: ignore
        return result

    def __reduce__(
        self,
    ) -> Tuple[Type["_LazyModule"], Tuple[str, str, Dict[str, List[str]]]]:
        return self.__class__, (self.__name__, self.__file__, self._import_structure)

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._import_structure:
            value = self._get_module(name)
        elif name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        """Imports the specified module ``module_name`` relative to this module.

        Args:
            module_name (str): The module to import.

        Returns:
            (ModuleType): The imported module ``module_name``.

        Raises:
            RuntimeError: The specified module ``module_name`` is not found.
        """
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as error:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the "
                f"following error (look up to see its traceback):\n{error}"
            ) from error
