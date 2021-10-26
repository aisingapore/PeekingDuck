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

"""
Python package requirements checker
"""

import collections
import logging
from pathlib import Path
import subprocess
from typing import Iterator, TextIO, Tuple, Union

import pkg_resources as pkg

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
PKD_REQ_TYPE_LEN = 6  # string length of either PYTHON or SYSTEM
PKD_REQ_TYPE_PYTHON = "PYTHON"  # type specifier for Python packages
ROOT = Path(__file__).resolve().parents[1]

OptionalRequirement = collections.namedtuple("OptionalRequirement", "name type")

def check_requirements(
        identifier: str,
        requirements_path: Path = ROOT / "optional_requirements.txt") -> int:
    """Checks if the packages specified by the `identifier` in the requirements
    file at `requirements_path` are present on the system. If `install` is True,
    attempts to install the packages

    Args:
        identifier (str): A unique identifier, typically a pipeline node name,
            used to specify which packages to check for
        requirements_path (Path): Path to the requirements file
    """
    with open(requirements_path) as infile:
        requirements = list(parse_requirements(infile, identifier))

    n_update = 0
    for req in requirements:
        if req.type == PKD_REQ_TYPE_PYTHON:
            try:
                pkg.require(req.name)
            except (pkg.DistributionNotFound, pkg.VersionConflict):
                logger.info("%s not found and is required, attempting "
                            "auto-update...", req.name)
                try:
                    logger.info(subprocess.check_output(
                        ["pip", "install", req.name]).decode())
                    n_update += 1
                except subprocess.CalledProcessError as exception:
                    logger.error(exception)
                    raise
        else:
            logger.warning("The %s node requires %s which needs to be "
                           "manually installed. Please follow the instructions "
                           "at %s and rerun. Ignore this warning if the "
                           "package is already installed", identifier,
                           req.name.strip(), "<install faq link>")

    if n_update > 0:
        logger.warning("%d package%s updated. Please rerun for the updates to "
                       "take effect.", n_update, "s" * int(n_update > 1))

    return n_update


def parse_requirements(file: TextIO,
                       identifier: str) -> Iterator[OptionalRequirement]:
    """Yield `pkg.Requirement` objects for each specification in `strings`

    `strings` must be a string, or a (possibly-nested) iterable thereof.
    """
    # create a steppable iterator, so we can handle \-continuations
    lines = iter(yield_lines(file, identifier))

    for line in lines:
        # Drop comments -- a hash without a space may be in a URL.
        if " #" in line:
            line = line[:line.find(" #")]
        req_type, req_name = split_type_and_name(line)
        if req_type == PKD_REQ_TYPE_PYTHON:
            req = pkg.Requirement(req_name)  # type: ignore
            requirement = OptionalRequirement(f"{req.name}{req.specifier}",
                                              req_type)
        else:
            requirement = OptionalRequirement(req_name, req_type)
        yield requirement


def yield_lines(strings: Union[TextIO, str], identifier: str) -> Iterator[str]:
    """Yield non-empty/non-comment lines of a string or sequence"""
    prefix = f"{identifier} "
    if isinstance(strings, str):
        for string in strings.splitlines():
            string = string.strip()
            # Return only optional requirement lines
            if string and string.startswith(prefix):
                yield string[len(prefix):]
    else:
        for string_item in strings:
            for string in yield_lines(string_item, identifier):
                yield string

def split_type_and_name(string: str) -> Tuple[str, str]:
    """Split an optional requirement line into the requirement type and name"""
    return string[:PKD_REQ_TYPE_LEN], string[PKD_REQ_TYPE_LEN:]
