# Copyright 2021 AI Singapore
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

import collections
import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO, Tuple, Union

import pkg_resources as pkg

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
PKD_NODE_PREFIX = "peekingduck.pipeline.nodes."
PKD_REQ_TYPE_LEN = 6  # string length of either PYTHON or SYSTEM
PKD_REQ_TYPE_PYTHON = "PYTHON"  # type specifier for Python packages
ROOT = Path(__file__).resolve().parents[1]

OptionalRequirement = collections.namedtuple("OptionalRequirement", "name type")


class RequirementChecker(importlib.abc.MetaPathFinder):
    """Checks for optional requirements from imports.

    While inheriting from MetaPathFinder is not strictly necessary, it serves
    as a reference for the required interface.
    """

    n_update = 0

    @staticmethod
    def find_spec(fullname: str, *_: Any) -> None:
        """Checks if the peekingduck.pipeline.nodes module being imported
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


def check_requirements(
    identifier: str, requirements_path: Path = ROOT / "optional_requirements.txt"
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
    with open(requirements_path) as infile:
        requirements = list(_parse_requirements(infile, identifier))

    n_update = 0
    for req in requirements:
        if req.type == PKD_REQ_TYPE_PYTHON:
            try:
                pkg.require(req.name)
            except (pkg.DistributionNotFound, pkg.VersionConflict):
                logger.info(
                    f"{req.name} not found and is required, attempting auto-update..."
                )
                try:
                    logger.info(
                        subprocess.check_output(["pip", "install", req.name]).decode()
                    )
                    n_update += 1
                except subprocess.CalledProcessError as exception:
                    logger.error(exception)
                    raise
        else:
            logger.warning(
                f"The {identifier} node requires {req.name.strip()} which needs to be "
                "manually installed. Please follow the instructions at "
                "https://peekingduck.readthedocs.io/en/stable/peekingduck.pipeline.nodes.html "
                "and rerun. Ignore this warning if the package is already installed"
            )

    if n_update > 0:
        logger.warning(
            f"{n_update} package{'s' * int(n_update > 1)} updated. Please rerun for "
            "the updates to take effect."
        )

    return n_update


def _parse_requirements(file: TextIO, identifier: str) -> Iterator[OptionalRequirement]:
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
        # Drop comments -- a hash without a space may be in a URL.
        if " #" in line:
            line = line[: line.find(" #")]
        req_type, req_name = _split_type_and_name(line)
        if req_type == PKD_REQ_TYPE_PYTHON:
            req = pkg.Requirement(req_name)  # type: ignore
            requirement = OptionalRequirement(f"{req.name}{req.specifier}", req_type)
        else:
            requirement = OptionalRequirement(req_name, req_type)
        yield requirement


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


def _split_type_and_name(string: str) -> Tuple[str, str]:
    """Split an optional requirement line into the requirement type and
    name.
    """
    return string[:PKD_REQ_TYPE_LEN], string[PKD_REQ_TYPE_LEN:]
