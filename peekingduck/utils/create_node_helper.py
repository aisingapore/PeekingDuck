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

"""Helper functions for the create node CLI command."""

import functools
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import click
import yaml

# Master map file for class name to object IDs for object detection models
MASTER_MAP = "pipeline/nodes/model/master_map.yml"

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def create_config_and_script_files(created_paths: Dict[str, Path]) -> None:
    """Creates a config and script file at the provided paths. The contents of
    these files will be copied from the template config and script file.
    """
    created_paths["config"].parent.mkdir(parents=True, exist_ok=True)
    created_paths["script"].parent.mkdir(parents=True, exist_ok=True)
    template_paths = get_config_and_script_paths(
        Path(__file__).resolve().parents[1],
        "configs",
        ("pipeline", "nodes"),
        "node_template",
    )
    with open(template_paths["config"]) as template_file, open(
        created_paths["config"], "w"
    ) as outfile:
        outfile.write(template_file.read())
    with open(template_paths["script"]) as template_file, open(
        created_paths["script"], "w"
    ) as outfile:
        lines = template_file.readlines()
        start = -1
        for i, line in enumerate(lines):
            if line.startswith('"'):
                start = i
                break
        # In case we couldn't find a starting line
        start = max(0, start)
        outfile.writelines(lines[start:])


def ensure_relative_path(node_subdir: str) -> str:
    """Checks that the subdir path does not contain parent directory
    navigator (..), is not absolute, and is not "peekingduck/pipeline/nodes".
    """
    pkd_node_subdir = "peekingduck/pipeline/nodes"
    if ".." in node_subdir:
        raise click.exceptions.UsageError("Path cannot contain '..'!")
    if Path(node_subdir).is_absolute():
        raise click.exceptions.UsageError("Path cannot be absolute!")
    if node_subdir == pkd_node_subdir:
        raise click.exceptions.UsageError(f"Path cannot be '{pkd_node_subdir}'!")
    return node_subdir


def ensure_valid_name(node_dir: Path, node_type: str, node_name: str) -> str:
    """Checks the validity of the specified node_name. Also checks if it
    already exists.
    """
    if re.match(r"^[a-zA-Z][\w\-]*[^\W_]$", node_name) is None:
        raise click.exceptions.UsageError("Invalid node name!")
    if (node_dir / node_type / f"{node_name}.py").exists():
        raise click.exceptions.UsageError("Node name already exists!")
    return node_name


def ensure_valid_name_partial(node_dir: Path, node_type: str) -> Callable:
    """Partial function to ensure_valid_name to provide a function that matches
    function signature required by ``value_proc`` in ``click.prompt()``.
    """
    return functools.partial(ensure_valid_name, node_dir, node_type)


def ensure_valid_type(node_type_choices: click.Choice, node_type: str) -> str:
    """Uses click's convert_type function to check the validity of the
    specified node_type. Re-raises with a custom error message to ensure
    consistency across click versions.
    """
    try:
        click.types.convert_type(node_type_choices)(node_type)
    except click.BadParameter:
        raise click.exceptions.UsageError(
            f"'{node_type}' is not one of {', '.join(map(repr, node_type_choices.choices))}."
        ) from None
    return node_type


def ensure_valid_type_partial(node_type_choices: click.Choice) -> Callable:
    """Partial function to ensure_valid_type to provide a function that matches
    the function signature required by ``value_proc`` in ``click.prompt()``.
    """
    return functools.partial(ensure_valid_type, node_type_choices)


def get_config_and_script_paths(
    parent_dir: Path,
    config_subdir: Union[str, Tuple[str, ...]],
    script_subdir: Union[str, Tuple[str, ...]],
    file_stem: str,
) -> Dict[str, Path]:
    """Returns the node config file and its corresponding script file."""
    if isinstance(config_subdir, tuple):
        config_subpath = Path(*config_subdir)
    else:
        config_subpath = Path(config_subdir)
    if isinstance(script_subdir, tuple):
        script_subpath = Path(*script_subdir)
    else:
        script_subpath = Path(script_subdir)

    return {
        "config": parent_dir / config_subpath / f"{file_stem}.yml",
        "script": parent_dir / script_subpath / f"{file_stem}.py",
    }


def obj_det_load_class_id_mapping(node_name: str) -> Dict[str, int]:
    """Loads class name to object ID mapping from the file
    peekingduck/pipeline/nodes/model/master_map.yml
    for object detection models only.

    Tech Notes
        master_map.yml comprises two documents:
        1. model mapping: tells which mapping system a particular object detection model uses
        2. class name to ID mapping: maps class name to ID, supports multiple mapping systems

    Args:
        node_name (str): Tells function which mapping to load,
                            Possible values = { model.efficientdet, model.yolo, model.yolox }.

    Returns:
        Dict[str, int]: Mapping of class names to object IDs relevant to given node_name
    """
    # use __file__ instead of self._base_dir as latter can be set to any (temp) path
    # without `peekingduck/` subdirectory, resulting in master map file not found error
    master_map_file = Path(__file__).parents[1] / MASTER_MAP
    # read both documents from master_map.yml
    with master_map_file.open() as map_file:
        model_map_dict, class_id_map_dict = yaml.safe_load_all(map_file)

    # node_name sanity check, to preempt non-object detection model nodes
    node_name_list = list(map(lambda x: f"model.{x}", model_map_dict.keys()))
    assert (
        node_name in node_name_list
    ), f"Node Name Error: expect one of {node_name_list} but got {node_name}"

    # determine mapping system to use based on given model
    model = node_name.replace("model.", "")
    model_map = model_map_dict[model]

    # construct class name->object ID map for the model
    class_id_map = {}
    for key, val in class_id_map_dict.items():
        class_id = val[model_map]
        class_id_map[key.lower()] = class_id
    return class_id_map


def obj_det_change_class_name_to_id(
    node_name: str, key: str, value: List[Any]
) -> Tuple[str, List[int]]:
    """Process object detection model node's detect_ids key and check for
    any class names to be converted to object IDs.
    E.g. person to 0, car to 2

    Args:
        node_name (str): to determine which object detection model is being used
                         because different models can use different object IDs.
        key (str): expected to be "detect_ids"; error otherwise.
        value (List[Any]): list of class names or object IDs for detection.
                           If object IDs, do nothing.
                           If class names, convert to object IDs.

    Returns:
        Tuple[str, List[int]]: "detect_ids", list of sorted object IDs.
    """
    class_id_map = obj_det_load_class_id_mapping(node_name)

    if not value:
        logger.warning("detect_ids list is empty, defaulting to detect person")
        value = ["person"]
    elif value == ["*"]:
        logger.info("Detecting all object classes")
        value = [*class_id_map]

    value_lc = [x.lower() if isinstance(x, str) else x for x in value]

    # parse value_lc for possible class name errors
    invalid_class_names = []
    for class_name in value_lc:
        if isinstance(class_name, str) and class_name not in class_id_map:
            invalid_class_names.append(class_name)

    if invalid_class_names:
        logger.warning(f"Invalid class names: {invalid_class_names}")

    # convert class names to numeric object IDs, any errors default to zero
    obj_ids_set = {
        x if isinstance(x, int) else class_id_map.get(x, 0) for x in value_lc
    }
    obj_ids_sorted_list = sorted(list(obj_ids_set))
    return key, obj_ids_sorted_list


def verify_option(value: Optional[str], value_proc: Callable) -> Optional[str]:
    """Verifies that input value via click.option matches the expected value.

    This sets ``value`` to ``None`` if it is invalid so the rest of the prompt
    can flow smoothly.

    Args:
        value (Optional[str]): Input value.
        value_proc (Callable): A function to check the validity of ``value``.

    Returns:
        (Optional[str]): ``value`` if it is a valid value. ``None`` if it is
            not.

    Raises:
        click.exceptions.UsageError: When ``value`` is invalid.
    """
    if value is None:
        return value
    try:
        value = value_proc(value)
    except click.exceptions.UsageError as error:
        click.echo(f"Error: {error.message}", err=True)
        value = None
    return value
