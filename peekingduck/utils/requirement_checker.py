import logging
from pathlib import Path
import subprocess

import pkg_resources as pkg

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
ROOT = Path(__file__).resolve().parents[2]

def check_requirements(
    identifier, requirements_path=ROOT / "requirements.txt", install=True
):
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
                        f"pip install '{req}'", shell=True).decode())
                    n_update += 1
                except subprocess.CalledProcessError as e:
                    logger.error(e)
                    raise
            else:
                logger.warn("%s. Please install and rerun.", msg)

    if n_update > 0:
        logger.warn("%d package%s updated. Please rerun for the updates to "
                    "take effect.", n_update, "s" * int(n_update > 1))


def parse_requirements(strs, identifier):
    """Yield `pkg.Requirement` objects for each specification in `strs`

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    # create a steppable iterator, so we can handle \-continuations
    lines = iter(yield_lines(strs, identifier))

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
        yield pkg.Requirement(line)


def yield_lines(strs, identifier):
    """Yield non-empty/non-comment lines of a string or sequence"""
    prefix = f"# OTF_req {identifier} "
    if isinstance(strs, str):
        for s in strs.splitlines():
            s = s.strip()
            # Return only OTF_req lines
            if s and s.startswith(prefix):
                yield s[len(prefix):]
    else:
        for ss in strs:
            for s in yield_lines(ss, identifier):
                yield s
