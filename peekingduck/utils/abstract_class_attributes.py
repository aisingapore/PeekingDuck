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

"""Abstract class attributes decorator."""

from typing import Any, Callable


def abstract_class_attributes(*names: str) -> Callable:
    """Class decorator to add abstract class attributes.

    Args:
        *names (str): The attribute names to be added.
    """

    def wrap_class(cls: object, *names: str) -> object:
        """Extends __init_subclass__."""
        for name in names:
            setattr(cls, name, NotImplemented)

        orig_init_subclass = cls.__init_subclass__

        def new_init_subclass(cls: object, **kwargs: Any) -> None:
            """Checks that the attributes are implemented."""
            try:
                # custom __init_subclass__ takes in a positional argument
                orig_init_subclass(cls, **kwargs)  # type: ignore
            except TypeError:
                # default does not
                orig_init_subclass(**kwargs)  # type: ignore

            # Check that the attributes are implemented
            for name in names:
                if getattr(cls, name, NotImplemented) is NotImplemented:
                    raise NotImplementedError(f"`{name}` needs to be implemented.")

        cls.__init_subclass__ = classmethod(new_init_subclass)  # type: ignore

        return cls

    return lambda cls: wrap_class(cls, *names)
