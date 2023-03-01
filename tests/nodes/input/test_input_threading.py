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

import errno
import os
import subprocess
from time import perf_counter

import pytest
import yaml

from tests.conftest import PKD_DIR

PKD_ROOT_DIR = PKD_DIR.parent
PIPELINE_PATH = PKD_ROOT_DIR / "threading_pipeline_config.yml"
# collect list of public RTSP URLs
RTSP_URL_GERMANY = (
    "http://clausenrc5.viewnetcam.com:50003/nphMotionJpeg?Resolution=320x240"
)
RTSP_URL_JAPAN_1 = "http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30"
RTSP_URL_JAPAN_2 = (
    "http://honjin1.miemasu.net/nphMotionJpeg?Resolution=640x480&Quality=Standard"
)
URL_LIST = [RTSP_URL_JAPAN_1, RTSP_URL_JAPAN_2, RTSP_URL_GERMANY]

# Helper Functions
def get_fps_number(avg_fps_msg: str) -> float:
    """Decodes FPS number from given "Avg FPS" string output by Peeking Duck
    in the format: "... Avg FPS over last n frames: x.xx ..."

    Args:
        avg_fps_msg (str): Peeking Duck's average FPS message string

    Returns:
        float: Frames per second number
    """
    # split "... Avg FPS over last n frames: x.xx ..."
    avg_fps_toks = list(avg_fps_msg.split("frames:"))
    # relevant FPS string segment in last token
    fps_toks = list(avg_fps_toks[-1].split(" "))
    # decode FPS value found in second token
    avg_fps = float(fps_toks[1])
    return avg_fps


@pytest.fixture
def pipeline_instance():
    yield
    try:
        os.remove(PIPELINE_PATH)
    except OSError as error:
        if error.errno != errno.ENOENT:
            # re-raise if exception isn't "no such file or directory"
            raise


# Unit Tests
@pytest.mark.usefixtures("pipeline_instance")
def test_input_threading():
    """Run input threading unit test.

    This test will do the following:
    1. Run input live test 1 without threading with threading_pipeline_config.yml file
       The test comprises input.visual, model.yolo and dabble.fps
    2. Run input live test 2 with threading with threading_pipeline_config.yml file
       The test comprises input.visual, model.yolo and dabble.fps
    3. Check average FPS from 2 is higher than 1
    """

    def run_rtsp_test(url: str, threading: bool) -> float:
        """Run input live test with given configuration and returns average FPS

        Args:
            url (str): link to video source, e.g. RTSP feed or mp4 file
            threading (bool): whether to enable threading or not

        Returns:
            float: average FPS
        """

        # create custom test config yml file
        nodes = {
            "nodes": [
                {
                    "input.visual": {
                        "source": url,
                        "threading": threading,
                    }
                },
                "model.yolo",
                "dabble.fps",
            ]
        }
        with open(PIPELINE_PATH, "w") as outfile:  # make new unit test config yml
            yaml.dump(nodes, outfile, default_flow_style=False)

        # run input live test
        num_sec = 30  # to run test for 30 seconds max
        avg_fps = 0
        # Technotes 2022-06-20:
        # previous `cmd = ["python", PKD_ROOT_DIR.name]` and `.Popen(... cwd=PKD_RUN_DIR, ...)`
        # breaks on Linux when PeekingDuck changes current working directory via full config path
        cmd = ["python", "__main__.py", "--config_path", PIPELINE_PATH.name]
        proc = subprocess.Popen(
            cmd,
            cwd=PKD_ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        st = perf_counter()
        while True:
            output = proc.stdout.readline()
            outstr = output.decode("utf-8")
            if "Avg FPS" in outstr:
                avg_fps = get_fps_number(outstr)
                break
            et = perf_counter()
            dur = et - st
            if dur > num_sec:
                break
        proc.kill()
        os.remove(PIPELINE_PATH)  # delete unit test yml

        return avg_fps

    def run_url_test(the_url: str) -> bool:
        # show test config
        print(f"url={the_url}")
        print(f"PKD_ROOT_DIR={PKD_ROOT_DIR}")
        print(f"PIPELINE_PATH={PIPELINE_PATH}")

        print("Run test without threading")
        avg_fps_1 = run_rtsp_test(url=the_url, threading=False)
        print("Run test with threading")
        avg_fps_2 = run_rtsp_test(url=the_url, threading=True)
        # check outcome
        res = avg_fps_2 > avg_fps_1
        print(f"avg_fps_1={avg_fps_1}, avg_fps_2={avg_fps_2}, res={res}")
        return res

    for url in URL_LIST:
        result = run_url_test(url)
        print(f"result={result}")
        # Early exit if a test case passes
        if result:
            return
    # None of the test cases passed, trigger test failure
    assert False
