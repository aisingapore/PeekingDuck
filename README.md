# PeekingDuck

PeekingDuck is a python framework dealing with model inference.

This toolkit provides state of the art computer vision models to make real time inference easy: *social distancing, people counting, license plate recognition etc.*. Customisability of the node modules ensures flexibility for unique usecases. Attached API server enables real time access to data analysis of the model pipe.

## Features

### Models

- Yolov4
- EfficientDet
- Blazepose

### Use Cases

- Social Distancing
- People Counting
- License Plate Recognition
- Vehicle Counting
- Privacy Filter

## How to Use for Developers (Temporary)

- `git clone` this repo
- `cd PeekingDuck`
- `pip install .`
- Choose the required nodes in [run_config.yml](run_config.yml)
- To run:
    ```
    peekingduck run --config_path <path_to_config>
    ```
- To create a new node, check out [CONTRIBUTING.md](CONTRIBUTING.md)




## Installation (WIP)

Use python package manager (pip) to install PeekingDuck

`pip install pkdk`

## Usage (WIP)

### Start New Projects with `init`
For new projects, we suggest to use the PeekingDuck cookiecutter starter:

```bash
> mkdir <new_project>
> cd <new_project>
> peekingduck init
```

### `get-configs`
Unless specified, all nodes in `peekingduck` will use the default configs for every node. To view and change these configs, you can use:

``` bash
> peekingduck get-configs
```

### `run`
You can run the PeekingDuck runner via the following command:

```bash
peekingduck run --config_path <path_to_config>
```

If `config_path` is not provided, this command will look for `run_config.yml` in the current directory.


For specific information on how to use peekingduck-cli, you can use `peekingduck --help`.

## Contributing (WIP)

We welcome contributions to the repository through pull requests. When making contributions, first create an issue to describe the problem and the intended changes.

Please note that we have a code of conduct for contributions to the repository.

## License

Licensed under [Apache License, Version 2.0](LICENSE)

