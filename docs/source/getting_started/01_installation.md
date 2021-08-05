## Install and Run PeekingDuck

1. Install PeekingDuck from [PyPI](https://pypi.org/project/peekingduck/).
    ```
    > pip install peekingduck
    ```
    *Note: if installing on a device with an ARM processor such as a Raspberry Pi, include the `--no-dependencies` flag.*

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
