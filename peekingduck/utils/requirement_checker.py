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

import logging
from pathlib import Path
import subprocess
from typing import Iterator, TextIO, Union

import pkg_resources as pkg

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
ROOT = Path(__file__).resolve().parents[2]

def check_requirements(identifier: str,
                       requirements_path: Path = ROOT / "requirements.txt",
                       install: bool = True) -> None:
    """Checks if the packages specified by the `identifier` in the requirements
    file at `requirements_path` are present on the system. If `install` is True,
    attempts to install the packages

    Args:
        identifier (str): A unique identifier, typically a pipeline node name,
            used to specify which packages to check for
        requiremetns_path (Path): Path to the requirements file
        install (bool): Flag to specify missing packages should be installed
    """
    with open(requirements_path) as infile:
        requirements = [f"{req.name}{req.specifier}"
                        for req in parse_requirements(infile, identifier)]

    n_update = 0
    for req in requirements:
        try:
            pkg.require(req)
        except (pkg.DistributionNotFound, pkg.VersionConflict):
            msg = f"{req} not found and is required"
            if install:
                logger.info("%s, attempting auto-update...", msg)
                try:
                    # Put req in quotes to prevent >=ver from being intepreted
                    # as redirect
                    logger.info(subprocess.check_output(
                        ["pip", "install", req]).decode())
                    n_update += 1
                except subprocess.CalledProcessError as exception:
                    logger.error(exception)
                    raise
            else:
                logger.warning("%s. Please install and rerun.", msg)

    if n_update > 0:
        logger.warning("%d package%s updated. Please rerun for the updates to "
                       "take effect.", n_update, "s" * int(n_update > 1))


def parse_requirements(file: TextIO,
                       identifier: str) -> Iterator[pkg.Requirement]:
    """Yield `pkg.Requirement` objects for each specification in `strings`

    `strings` must be a string, or a (possibly-nested) iterable thereof.
    """
    # create a steppable iterator, so we can handle \-continuations
    lines = iter(yield_lines(file, identifier))

    for line in lines:
        # Drop comments -- a hash without a space may be in a URL.
        if " #" in line:
            line = line[:line.find(" #")]
        # If there is a line continuation, drop it, and append the next line.
        if line.endswith("\\"):
            line = line[:-2].strip()
            try:
                line += next(lines)
            except StopIteration:
                return
        yield pkg.Requirement(line)  # type: ignore


def yield_lines(strings: Union[TextIO, str], identifier: str) -> Iterator[str]:
    """Yield non-empty/non-comment lines of a string or sequence"""
    prefix = f"# OTF_req {identifier} "
    if isinstance(strings, str):
        for string in strings.splitlines():
            string = string.strip()
            # Return only OTF_req lines
            if string and string.startswith(prefix):
                yield string[len(prefix):]
    else:
        for string_item in strings:
            for string in yield_lines(string_item, identifier):
                yield string
