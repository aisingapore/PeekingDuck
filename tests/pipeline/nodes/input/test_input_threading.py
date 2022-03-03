import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import yaml

PKD_ROOT_DIR = Path(__file__).parents[4]  # dependent on __file__ location
PKD_CONFIG_ORIG_PATH = PKD_ROOT_DIR / "run_config.yml"
PKD_CONFIG_BAK_PATH = PKD_ROOT_DIR / "run_config_orig_bak.yml"
PKD_RUN_DIR = Path(__file__).parents[5]  # dependent on __file__ location
RTSP_URL = "http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30"

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


@contextmanager
def run_config_yml():
    """Save and restore current run_config.yml"""
    try:
        config_saved = False
        if os.path.isfile(PKD_CONFIG_ORIG_PATH):
            print("Backup existing run_config.yml")
            os.rename(src=PKD_CONFIG_ORIG_PATH, dst=PKD_CONFIG_BAK_PATH)
            config_saved = True
        yield
    finally:
        if config_saved:
            print("Restore backed up run_config.yml")
            os.rename(src=PKD_CONFIG_BAK_PATH, dst=PKD_CONFIG_ORIG_PATH)


# Unit Tests
def test_input_threading():
    """Run input threading unit test.

    This test will do the following:
    1. Backup original run_config.yml in Peeking Duck directory
    2. Run input live test 1 without threading with custom run_config.yml file
       The test comprises input.live, model.yolo and dabble.fps
    3. Run input live test 2 with threading with custom run_config.yml file
       The test comprises input.live, model.yolo and dabble.fps
    4. Restore original run_config.yml
    5. Check average FPS from 2 is higher than 1
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
                    "input.live": {
                        "input_source": url,
                        "threading": threading,
                    }
                },
                "model.yolo",
                "dabble.fps",
            ]
        }
        with open(
            PKD_CONFIG_ORIG_PATH, "w"
        ) as outfile:  # make new unit test config yml
            yaml.dump(nodes, outfile, default_flow_style=False)

        # run input live test
        num_sec = 60  # to run test for 60 seconds max
        avg_fps = 0
        cmd = ["python", PKD_ROOT_DIR.name]
        proc = subprocess.Popen(
            cmd,
            cwd=PKD_RUN_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
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
        os.remove(PKD_CONFIG_ORIG_PATH)  # delete unit test yml

        return avg_fps

    res = False
    with run_config_yml():
        print("Run test without threading")
        avg_fps_1 = run_rtsp_test(url=RTSP_URL, threading=False)

        print("Run test with threading")
        avg_fps_2 = run_rtsp_test(url=RTSP_URL, threading=True)

        res = avg_fps_2 > avg_fps_1
        print(f"avg_fps_1={avg_fps_1}, avg_fps_2={avg_fps_2}, res={res}")
    assert res
