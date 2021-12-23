# Copyright 2021 AI Singapore
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
#
# Original copyright (c) 2018 Kaiyang Zhou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Additional functions for pytorch model inference.
"""

from typing import Any, Dict
from collections import OrderedDict
from functools import partial
from pathlib import Path
import pickle
import warnings
import torch
import torch.nn as nn


def check_isfile(fpath: str) -> bool:
    """Checks if the given path is a file.

    Args:
        fpath (str): File path.

    Returns:
       bool: True/False if file path exists.
    """
    isfile = Path(fpath).is_file()
    if not isfile:
        warnings.warn(f"No file found at {fpath}")
    return isfile


def load_checkpoint(fpath: str) -> Dict[Any, Any]:
    """Loads model checkpoint.

    Args:
        fpath (str): Path to checkpoint.

    Raises:
        ValueError: File path is not provided.
        FileNotFoundError: File is not found at path.

    Returns:
        Dict[Any, Any]: Model checkpoints.
    """
    if fpath is None:
        raise ValueError("File path is None")

    fpath = Path(fpath).resolve()
    if not Path(fpath).exists():
        raise FileNotFoundError(f"File is not found at {fpath}")

    map_location = None if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")  # type: ignore
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print(f"Unable to load checkpoint from {fpath}")
        raise

    return checkpoint


def load_pretrained_weights(model: nn.Module, weight_path: str) -> None:
    """Loads pretrained weights to model.

    Features:
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): Model network.
        weight_path (str): Path to pretrained weights.
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  # discard module.

        if key in model_dict and model_dict[key].size() == value.size():
            new_state_dict[key] = value
            matched_layers.append(key)
        else:
            discarded_layers.append(key)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if not matched_layers:
        warnings.warn(
            f"The pretrained weights {weight_path} cannot be loaded, "
            "please check the key names manually (** ignored and continue **)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if discarded_layers:
            print(
                "** The following layers are discarded due to unmatched"
                f"keys or layer size:\n{discarded_layers}\n"
                "This action is normal."
            )
