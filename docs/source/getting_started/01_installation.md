# Install and Run PeekingDuck

1. Install PeekingDuck from [PyPI](https://pypi.org/project/peekingduck/).
    ```
    > pip install peekingduck
    ```
    *Note: if installing on a ARM-based device such as a Raspberry Pi or M1 MacBook, include the `--no-dependencies` flag, and separately install other dependencies listed in our [requirements.txt](https://github.com/aimakerspace/PeekingDuck/blob/dev/requirements.txt). See our guide for [M1 Mac](https://peekingduck.readthedocs.io/en/stable/getting_started/01_installation.html#m1-mac-installation) installation.*

2. Create a project folder at a convenient location, and initialize a PeekingDuck project.
    ```
    > mkdir <project_dir>
    > cd <project_dir>
    > peekingduck init
    ```
    The following files and folders will be created upon running `peekingduck init`:
    - `run_config.yml` is the main configuration file for PeekingDuck. It currently contains the [default configuration](https://github.com/aimakerspace/PeekingDuck/blob/dev/run_config.yml), and we'll show you how to modify it in the [configuration guide](../getting_started/02_configure_pkdk.md).
    - `custom_nodes` is an optional feature that is discussed in the [custom nodes guide](../getting_started/03_custom_nodes.md).
    ```
    <project_dir>
     ├── run_config.yml
     └── src
          └── custom_nodes
               └── configs
    ```

3. Run a demo.
    ```
    > peekingduck run
    ```

    If you have a webcam, you should see the demo running live:

    <img src="https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/readme/yolo_demo.gif" width="50%">

    The previous command looks for a `run_config.yml` in the current directory. You can also specify the path of a different config file to be used, as follows:
    ```
    > peekingduck run --config_path <path_to_config>
    ```

    Terminate the program by clicking on the output screen and pressing `q`.

4. For more help on how to use PeekingDuck's command line interface, you can use `peekingduck --help`.


## M1 Mac Installation

Apple started releasing Macs with their proprietary [M1](https://en.wikipedia.org/wiki/Apple_M1) ARM-based chip in late 2020, a significant change from the previous Intel processors. We've successfully tested PeekingDuck on a few M1 Macs with these steps:

Prerequisites
- Install [homebrew](https://brew.sh/)
- Install miniforge using homebrew: `brew install miniforge`

Install PeekingDuck in Conda Environment for MacOS Big Sur 11.x
```
conda create -n pkd38 python=3.8
conda activate pkd38
conda install click colorama opencv openblas pyyaml requests scipy shapely tqdm
conda install -c apple tensorflow-deps=2.6.0
pip install tensorflow-estimator==2.6.0 tensorflow-macos==2.6.0
pip install tensorflow-metal==0.2.0
pip install opencv-contrib-python
pip install peekingduck --no-dependencies
```

Notes:
- Only Python 3.8 is available for Conda on M1 Mac - Python 3.6 or 3.7 are not available
- Apple's tensorflow will install numpy 1.19.5, which will get upgraded by opencv-contrib-python to 1.21
- Todo: Add installation instructions for `pytorch` and `torchvision`

If this doesn't work for you, do check out our [issues](https://github.com/aimakerspace/PeekingDuck/issues) to see if the community of M1 Mac users have alternative solutions. We will update these instructions as we get more feedback.
