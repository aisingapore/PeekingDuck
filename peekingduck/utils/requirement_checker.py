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

"""Python package requirements checker."""

import importlib
import locale
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, List, Optional, TextIO, Tuple, Union

import pkg_resources as pkg

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
PKD_NODE_PREFIX = "peekingduck.nodes."
PKD_REQ_TYPE_LEN = 6  # string length of either PYTHON or SYSTEM
PKD_REQ_TYPE_PYTHON = "PYTHON"  # type specifier for Python packages
ROOT = Path(__file__).resolve().parents[1]


class RequirementChecker(importlib.abc.MetaPathFinder):
    """Checks for optional requirements from imports.

    While inheriting from MetaPathFinder is not strictly necessary, it serves
    as a reference for the required interface.
    """

    n_update = 0

    @staticmethod
    def find_spec(fullname: str, *_: Any) -> None:
        """Checks if the peekingduck.nodes module being imported
        contains optional requirements. Attempt to install if it does.

        Args:
            fullname (:obj:`str`): Name of the module being imported.
        """
        if fullname.startswith(PKD_NODE_PREFIX):
            try:
                RequirementChecker.n_update += check_requirements(
                    fullname[len(PKD_NODE_PREFIX) :]
                )
            except subprocess.CalledProcessError:
                sys.exit(1)


class OptionalRequirement:
    """Stores information about an optional requirement. Includes alternative
    packages if specified.
    """

    def __init__(self, name: str, req_type: str) -> None:
        self.name = name
        self.type = req_type
        self.packages = [name]

    def add_alternative(self, name: str) -> None:
        """Append a package name to `alternatives`."""
        self.packages.append(name)

    @classmethod
    def from_pkg_resources(
        cls, requirements: List[pkg.Requirement], req_type: str
    ) -> "OptionalRequirement":
        """Creates an OptionalRequirement object from a list of
        pkg_resources.Requirement. Appends extra package names as
        `alternatives` if there is more than one element in the list.
        """
        optional_requirement = cls(
            f"{requirements[0].name}{requirements[0].specifier}", req_type
        )
        for req in requirements[1:]:
            optional_requirement.add_alternative(f"{req.name}{req.specifier}")

        return optional_requirement


def check_requirements(
    identifier: str,
    requirements_path: Path = ROOT / "optional_requirements.txt",
    flags: Optional[str] = None,
) -> int:
    """Checks if the packages specified by the ``identifier`` in the
    requirements file at ``requirements_path`` are present on the system. If
    ``install`` is ``True``, attempts to install the packages.

    Args:
        identifier (:obj:`str`): A unique identifier, typically a pipeline node
            name, used to specify which packages to check for.
        requirements_path (Path): Path to the requirements file

    Returns:
        (:obj:`int`): The number of packages updated.
    """
    with open(requirements_path, encoding=locale.getpreferredencoding(False)) as infile:
        requirements = list(_parse_requirements(infile, identifier, flags))

    n_update = 0
    for req in requirements:
        if req.type != PKD_REQ_TYPE_PYTHON:
            logger.warning(
                f"The {identifier} node requires {req.name.strip()} which needs to be "
                "manually installed. Please follow the instructions at "
                "https://peekingduck.readthedocs.io/en/stable/master.html#api-documentation "
                "and rerun. Ignore this warning if the package is already installed"
            )
            continue
        if _require_any(req):
            continue
        # Try to find a package that can be pip installed, raise a
        # subprocess.CalledProcessError with all the attempted commands if
        # none of the packages can be installed
        errors = []
        for package in req.packages:
            logger.info(
                f"{package} not found and is required, attempting auto-update..."
            )
            try:
                logger.info(
                    subprocess.check_output(["pip", "install", package]).decode()
                )
                n_update += 1
                break
            except subprocess.CalledProcessError as exception:
                errors.append(exception)
        else:  # no break
            logger.error("\n".join(map(str, errors)))
            raise errors[0]
    return n_update


def _parse_requirements(
    file: TextIO, identifier: str, flags: Optional[str]
) -> Iterator[OptionalRequirement]:
    """Yield ``OptionalRequirement`` objects for each specification in
    ``strings``.

    ``strings`` must be a string, or a (possibly-nested) iterable thereof.

    Arg:
        file (TextIO): The file object containing optional requirements.
        identifier (str): A unique identifier, typically a pipeline node name,
            used to specify which packages to check for.

    Returns:
        (Iterator[OptionalRequirements]): Optional requirements, both Python
            and system packages, specified under the unique identifier.
    """
    lines = iter(_yield_lines(file, identifier))
    for line in lines:
        skip, line = _select_line(flags, line)
        if skip:
            continue
        req_type, req_name = _split_type_and_name(line)
        if req_type == PKD_REQ_TYPE_PYTHON:
            reqs = [pkg.Requirement.parse(name) for name in req_name.split("|")]
            requirement = OptionalRequirement.from_pkg_resources(reqs, req_type)
        else:
            requirement = OptionalRequirement(req_name, req_type)
        yield requirement


def _select_line(flags: Optional[str], line: str) -> Tuple[bool, str]:
    """Selects the optional requirement line for further processing and
    installation. If flags are not provided, skips line with selector. If flags
    are provided, skips lines without selector or with wrong selector.

    Args:
        flags (Optional[str]): A string indicating the combination of configs
            used to select this package for installation.
        line (str): A line in the optional requirements file containing the
            package specification and optional comments/selector.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean flag indicating if the
        line should be skipped and a clean requirement line without any comments.
    """
    flags_start = "flags:"
    flags_stop = "$"
    comment_start = line.find(" #")  # only used when there is comment in line
    is_clean = " #" not in line  # no comment in line
    is_comment = not is_clean and flags_start not in line  # only comment in line

    if flags is None:
        if is_clean:
            return False, line
        if is_comment:
            return False, line[:comment_start]
        return True, line

    if is_clean or is_comment:
        return True, line
    if f"{flags_start} {flags}{flags_stop}" in line[comment_start:]:
        return False, line[:comment_start]
    return True, line


def _require_any(req: OptionalRequirement) -> bool:
    """Checks if the optional requirement or any of its alternatives is found
    in the current environment.

    Args:
        req (OptionalRequirement): The specified optional requirement.

    Returns:
        (bool): True if the optional requirement or any of its alternatives is
        found.
    """
    for package in req.packages:
        try:
            pkg.require(package)
            return True
        except (pkg.DistributionNotFound, pkg.VersionConflict):
            pass
    return False


def _split_type_and_name(string: str) -> Tuple[str, str]:
    """Split an optional requirement line into the requirement type and
    name."""
    return string[:PKD_REQ_TYPE_LEN], string[PKD_REQ_TYPE_LEN:]


def _yield_lines(strings: Union[TextIO, str], identifier: str) -> Iterator[str]:
    """Yield lines with ``identifier`` as the prefix.

    Args:
        strings (Union[TextIO, str]): Either a file object or a line from the
            file.
        identifier (str): A unique identifier, typically a pipeline node name,
            used to specify which packages to check for.

    Returns:
        (Iterator[str]): Lines with ``identifier`` as the prefix.
    """
    prefix = f"{identifier} "
    if isinstance(strings, str):
        for string in strings.splitlines():
            string = string.strip()
            # Return only optional requirement lines
            if string and string.startswith(prefix):
                yield string[len(prefix) :]
    else:
        for string_item in strings:
            for string in _yield_lines(string_item, identifier):
                yield string
