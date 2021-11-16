import os
import subprocess
from peekingduck.pipeline.nodes.input.recorded import Node

# Change to dir to use `python PeekingDuck` from CLI
TEST_IMG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../peekingduck/images/testing"
)
TEST_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_cli_config.yml"
)
PY_RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")


def create_reader():
    media_reader = Node(
        {
            "input": "source",
            "output": "img",
            "input_dir": TEST_IMG_DIR,
        }
    )
    return media_reader


def test_cli_help():
    os.chdir(PY_RUN_DIR)
    cmd = "python PeekingDuck --help"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (out, _) = proc.communicate()
    out_str = out.decode("utf-8")
    exit_status = proc.returncode
    assert "log-level" in str(out_str)
    assert exit_status == 0


def test_cli_run():
    os.chdir(PY_RUN_DIR)
    cmd = "python PeekingDuck"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    try:
        out, _ = proc.communicate(timeout=10)  # kill after 10 seconds
    except subprocess.TimeoutExpired:
        proc.terminate()
    exit_status = proc.returncode
    assert exit_status is None


def test_cli_run_log_level_debug():
    os.chdir(PY_RUN_DIR)
    cmd = f"python PeekingDuck --log-level debug --config_path {TEST_CONFIG_PATH}"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (out, _) = proc.communicate()
    out_str = out.decode("utf-8")
    exit_status = proc.returncode
    assert "debug" in str(out_str)
    assert exit_status == 0
