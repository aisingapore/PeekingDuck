import subprocess
from pathlib import Path
from unittest import TestCase, mock

import pytest

from peekingduck.utils.requirement_checker import (
    PKD_NODE_PREFIX,
    RequirementChecker,
    check_requirements,
)

INSTALL_FAQ_LINK = (
    "https://peekingduck.readthedocs.io/en/stable/peekingduck.pipeline.nodes.html"
)
PKG_REQ_TYPE_PYTHON = "PYTHON"
PKG_REQ_TYPE_SYSTEM = "SYSTEM"
NODE_WITH_UPDATE = "node_type.node_name0"
NODE_WITH_SYS_PKG = "node_type.node_name1"
PY_PKGS = [
    ["pkg_name0", "==", "ver0"],
    ["pkg_name1", ">=", "ver1"],
]
# Comments are added to get all branches of parse_requirements to run
REQUIREMENTS_CONTENT = [
    f"{NODE_WITH_UPDATE} {PKG_REQ_TYPE_PYTHON} pytest >= 6.2.3",
    f"{NODE_WITH_UPDATE} {PKG_REQ_TYPE_PYTHON} {' '.join(PY_PKGS[0])}",
    f"{NODE_WITH_UPDATE} {PKG_REQ_TYPE_PYTHON} {' '.join(PY_PKGS[1])}",
    f"{NODE_WITH_SYS_PKG} {PKG_REQ_TYPE_SYSTEM} sys_package_name0  # inline comment",
    f"{NODE_WITH_SYS_PKG} {PKG_REQ_TYPE_SYSTEM} sys_package_name1",
]


@pytest.fixture
def requirements_file():
    requirements_path = Path.cwd() / "optional_requirements.txt"
    with open(requirements_path, "w") as outfile:
        for line in REQUIREMENTS_CONTENT:
            outfile.write(f"{line}\n")
    return requirements_path


def replace_subprocess_check_output(args):
    # Need to be byte string since we are chaining .decode()
    return " ".join(args).encode()


@pytest.mark.usefixtures("tmp_dir")
class TestRequirementChecker:
    def test_update_failure(self, requirements_file):
        ret_code = 123
        cmd = "command"
        with mock.patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(ret_code, cmd),
        ), TestCase.assertLogs(
            "peekingduck.utils.requirement_checker.logger"
        ) as captured, pytest.raises(
            subprocess.CalledProcessError
        ):
            check_requirements(NODE_WITH_UPDATE, requirements_file)
            assert (
                f"Command '{cmd}' returned non-zero exit status {ret_code}."
                == captured.records[1].getMessage()
            )

    def test_checker_class_update_failure(self, requirements_file):
        ret_code = 123
        cmd = "command"
        with mock.patch(
            "peekingduck.utils.requirement_checker.check_requirements",
            side_effect=subprocess.CalledProcessError(ret_code, cmd),
        ), pytest.raises(SystemExit):
            RequirementChecker.find_spec(PKD_NODE_PREFIX + NODE_WITH_UPDATE)

    def test_node_with_python_updates(self, requirements_file):
        with mock.patch(
            "subprocess.check_output", wraps=replace_subprocess_check_output
        ), TestCase.assertLogs(
            "peekingduck.utils.requirement_checker.logger"
        ) as captured:
            # pytest >= 6.2.3 is guaranteed by cicd_requirements.txt so
            # only 2 updates
            assert check_requirements(NODE_WITH_UPDATE, requirements_file) == 2
            for i, record in enumerate(captured.records):
                idx = int(i / 2)
                if i == 4:
                    assert "2 packages updated" in record.getMessage()
                elif i % 2 == 0:
                    assert f"{''.join(PY_PKGS[idx])} not found" in record.getMessage()
                else:
                    assert f"pip install {''.join(PY_PKGS[idx])}" in record.getMessage()

    def test_node_with_system_packages(self, requirements_file):
        with TestCase.assertLogs(
            "peekingduck.utils.requirement_checker.logger"
        ) as captured:
            assert check_requirements(NODE_WITH_SYS_PKG, requirements_file) == 0
            for i, record in enumerate(captured.records):
                msg = record.getMessage()
                assert f"The {NODE_WITH_SYS_PKG} node" in msg
                assert f"requires sys_package_name{i}" in msg
                assert f"instructions at {INSTALL_FAQ_LINK}" in msg
