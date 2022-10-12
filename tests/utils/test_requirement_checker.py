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

import subprocess
from pathlib import Path
from unittest import TestCase, mock

import pytest

from peekingduck.utils.requirement_checker import (
    PKD_NODE_PREFIX,
    RequirementChecker,
    check_requirements,
)
from tests.conftest import assert_msg_in_logs

INSTALL_FAQ_LINK = (
    "https://peekingduck.readthedocs.io/en/stable/master.html#api-documentation"
)
PKG_REQ_TYPE_PYTHON = "PYTHON"
PKG_REQ_TYPE_SYSTEM = "SYSTEM"
NODE_WITH_UPDATE = "node_type.node_name0"
NODE_WITH_SYS_PKG = "node_type.node_name1"
PY_PKGS = [
    ["pkg_name0", "==", "ver0"],
    ["pkg_name1", ">=", "ver1"],
    ["pkg_name2", "==", "ver2"],
    ["pkg_name3", ">=", "ver3"],
    ["pkg_name4", ">=", "ver4"],
]
# Comments are added to get all branches of parse_requirements to run
REQUIREMENTS_CONTENT = [
    f"{NODE_WITH_UPDATE} {PKG_REQ_TYPE_PYTHON} pytest >= 6.2.3",
    f"{NODE_WITH_UPDATE} {PKG_REQ_TYPE_PYTHON} {' '.join(PY_PKGS[0])}",
    f"{NODE_WITH_UPDATE} {PKG_REQ_TYPE_PYTHON} {' '.join(PY_PKGS[1])}",
    f"{NODE_WITH_SYS_PKG} {PKG_REQ_TYPE_SYSTEM} sys_package_name0  # inline comment",
    f"{NODE_WITH_SYS_PKG} {PKG_REQ_TYPE_SYSTEM} sys_package_name1",
    f"{NODE_WITH_UPDATE} {PKG_REQ_TYPE_PYTHON} {' '.join(PY_PKGS[2])} | {' '.join(PY_PKGS[3])}",
    f"{NODE_WITH_UPDATE} {PKG_REQ_TYPE_PYTHON} {' '.join(PY_PKGS[4])} | pytest >= 6.2.3",
]


@pytest.fixture(name="requirements_file")
def fixture_requirements_file():
    requirements_path = Path.cwd() / "optional_requirements.txt"
    with open(requirements_path, "w") as outfile:
        for line in REQUIREMENTS_CONTENT:
            outfile.write(f"{line}\n")
    return requirements_path


@pytest.fixture(name="behavior_requirements_file")
def fixture_behavior_requirements_file(request):
    """Creates an optional requirements file which will trigger various
    checker behavior. The types of behavior is defined by `checker_behavior`.

    Args:
        request: The checker behavior to trigger which is a key of
            `checker_behavior`.
    """
    # Maps the requirement check behavior that is triggered when creating a
    # requirements up to this line number.
    checker_behavior = {
        "python": 3,
        "system": 6,
        "python + alt": 7,
        "python + alt short circuit": 8,
    }
    requirements_path = Path.cwd() / "optional_requirements.txt"
    with open(requirements_path, "w") as outfile:
        for line in REQUIREMENTS_CONTENT[: checker_behavior[request.param]]:
            outfile.write(f"{line}\n")
    return requirements_path


def replace_subprocess_check_output(args):
    # Need to be byte string since we are chaining .decode()
    return " ".join(args).encode()


@pytest.mark.usefixtures("tmp_dir")
class TestRequirementChecker:
    def test_check_requirements_reports_install_failure(self, requirements_file):
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
            assert_msg_in_logs(
                f"Command '{cmd}' returned non-zero exit status {ret_code}.",
                captured.records,
            )

    def test_checker_class_exits_the_programe_upon_update_failure(
        self, requirements_file
    ):
        ret_code = 123
        cmd = "command"
        with mock.patch(
            "peekingduck.utils.requirement_checker.check_requirements",
            side_effect=subprocess.CalledProcessError(ret_code, cmd),
        ), pytest.raises(SystemExit):
            RequirementChecker.find_spec(PKD_NODE_PREFIX + NODE_WITH_UPDATE)

    @pytest.mark.parametrize("behavior_requirements_file", ("python",), indirect=True)
    def test_only_update_missing_python_packages(self, behavior_requirements_file):
        with mock.patch(
            "subprocess.check_output", wraps=replace_subprocess_check_output
        ), TestCase.assertLogs(
            "peekingduck.utils.requirement_checker.logger"
        ) as captured:
            # pytest >= 6.2.3 is guaranteed by requirements_cicd.txt so
            # only 2 updates
            assert check_requirements(NODE_WITH_UPDATE, behavior_requirements_file) == 2

            assert_msg_in_logs(f"{''.join(PY_PKGS[0])} not found", captured.records)
            assert_msg_in_logs(f"{''.join(PY_PKGS[1])} not found", captured.records)
            assert_msg_in_logs(f"pip install {''.join(PY_PKGS[0])}", captured.records)
            assert_msg_in_logs(f"pip install {''.join(PY_PKGS[1])}", captured.records)

    @pytest.mark.parametrize("behavior_requirements_file", ("system",), indirect=True)
    def test_log_instructions_for_optional_system_packages(
        self, behavior_requirements_file
    ):
        with TestCase.assertLogs(
            "peekingduck.utils.requirement_checker.logger"
        ) as captured:
            assert (
                check_requirements(NODE_WITH_SYS_PKG, behavior_requirements_file) == 0
            )
            assert_msg_in_logs(f"The {NODE_WITH_SYS_PKG} node", captured.records)
            assert_msg_in_logs("requires sys_package_name0", captured.records)
            assert_msg_in_logs("requires sys_package_name1", captured.records)
            assert_msg_in_logs(f"instructions at {INSTALL_FAQ_LINK}", captured.records)

    @pytest.mark.parametrize(
        "behavior_requirements_file", ("python + alt",), indirect=True
    )
    def test_install_only_one_alternative(self, behavior_requirements_file):
        with mock.patch(
            "subprocess.check_output", wraps=replace_subprocess_check_output
        ), TestCase.assertLogs(
            "peekingduck.utils.requirement_checker.logger"
        ) as captured:
            # only installs one of the alternatives, so 3 updates
            assert check_requirements(NODE_WITH_UPDATE, behavior_requirements_file) == 3

            assert_msg_in_logs(f"{''.join(PY_PKGS[2])} not found", captured.records)
            assert_msg_in_logs(f"pip install {''.join(PY_PKGS[2])}", captured.records)

    @pytest.mark.parametrize(
        "behavior_requirements_file", ("python + alt short circuit",), indirect=True
    )
    def test_does_not_install_if_alternative_is_present(
        self, behavior_requirements_file
    ):
        with mock.patch(
            "subprocess.check_output", wraps=replace_subprocess_check_output
        ), TestCase.assertLogs(
            "peekingduck.utils.requirement_checker.logger"
        ) as captured:
            # doesn't install if one of the alternative is available
            assert check_requirements(NODE_WITH_UPDATE, behavior_requirements_file) == 3
