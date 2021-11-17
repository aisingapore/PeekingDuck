import os
import subprocess
from time import perf_counter
from pathlib import Path
import yaml

PKD_ROOT_DIR = Path(__file__).parents[4]  # dependent on __file__ location
PKD_CONFIG_ORIG = PKD_ROOT_DIR / "run_config.yml"
PKD_CONFIG_BAK = PKD_ROOT_DIR / "run_config_orig_bak.yml"
PKD_RUN_DIR = Path(__file__).parents[5]  # dependent on __file__ location

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


# Unit Tests
def test_input_threading():
    """Run input threading unit test.

    This test will do the following:
    1. Prepare unit test environment
    2. Backup original run_config.yml in Peeking Duck directory
    3. Run input live test 1 without threading with custom run_config.yml file
       The test comprises input.live, model.yolo and dabble.fps
    4. Run input live test 2 with threading with custom run_config.yml file
       The test comprises input.live, model.yolo and dabble.fps
    5. Restore original run_config.yml
    6. Check average FPS from 2 is higher than 1
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
        with open(PKD_CONFIG_ORIG, "w") as outfile:  # make new unit test config yml
            yaml.dump(nodes, outfile, default_flow_style=False)

        # run input live test
        num_sec = 60  # to run test for 60 seconds max
        avg_fps = 0
        cmd = f"python PeekingDuck"
        proc = subprocess.Popen(  # nosec
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, bufsize=1
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

        return avg_fps

    # 1. Setup unit test environment
    os.chdir(PKD_RUN_DIR)

    # 2. Backup original run_config.yml
    os.rename(src=PKD_CONFIG_ORIG, dst=PKD_CONFIG_BAK)  # backup config yml

    # 3. Run input live test without threading
    avg_fps_1 = run_rtsp_test(
        url="http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30",
        threading=False,
    )
    os.remove(PKD_CONFIG_ORIG)  # delete unit test yml

    # 4. Run input live test with threading
    avg_fps_2 = run_rtsp_test(
        url="http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30",
        threading=True,
    )
    os.remove(PKD_CONFIG_ORIG)  # delete unit test yml

    # 5. Restore original config
    os.rename(src=PKD_CONFIG_BAK, dst=PKD_CONFIG_ORIG)

    # 6. Check we get higher FPS for 2 than 1
    res = avg_fps_2 > avg_fps_1
    print(f"avg_fps_1={avg_fps_1}, avg_fps_2={avg_fps_2}, res={res}")
    assert res
