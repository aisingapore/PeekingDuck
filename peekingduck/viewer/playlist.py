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

"""Implements the PeekingDuck Pipeline PlayList class"""

from typing import Dict, List, Union
from datetime import datetime
from pathlib import Path
import logging
import yaml


# Globals
PKD_CONFIG_DIR = ".peekingduck"
PKD_PLAYLIST_FILE = "playlist.yml"


class PipelineStats:
    """Implements immutable PipelineStats class to store pipeline-related information."""

    def __init__(self, pipeline: str) -> None:
        self._hash = hash(pipeline)
        self._pipeline = pipeline
        pipeline_path = Path(pipeline)
        self._name = pipeline_path.name
        self._datetime = (
            pipeline_path.stat().st_mtime if pipeline_path.exists() else None
        )

    def __eq__(self, obj: "PipelineStats") -> bool:
        return self._pipeline == obj._pipeline

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, obj: "PipelineStats") -> bool:
        return (self._name, self._pipeline) < (obj._name, obj._pipeline)

    def __repr__(self) -> str:
        return self._pipeline

    @property
    def datetime(self) -> str:
        """Get last modified date/time of pipeline file.

        Returns:
            str: Last modified date/time string.
        """
        if self._datetime:
            return datetime.fromtimestamp(self._datetime).strftime("%Y-%m-%d-%H:%M:%S")
        return ""

    @property
    def name(self) -> str:
        """Get pipeline name.

        Returns:
            str: Name of pipeline.
        """
        return self._name

    @property
    def pipeline(self) -> str:
        """Get pipeline full path name.

        Returns:
            str: Full path name of pipeline.
        """
        return self._pipeline


class PlayList:
    """Implements the PlayList class to store pipelines in playlist and to handle
    the internal data structures for managing pipelines"""

    def __init__(self, home_path: Path) -> None:
        self.logger = logging.getLogger(__name__)
        # Construct path to ~user_home/.peekingduck/playlist.yaml
        self._playlist_dir = home_path / PKD_CONFIG_DIR
        self._playlist_dir.mkdir(exist_ok=True)
        self._playlist_path = self._playlist_dir / PKD_PLAYLIST_FILE
        self.logger.debug(f"playlist_path={self._playlist_path}")
        self.load_playlist_file()

    def __iter__(self) -> "PlayList":
        self._iter_idx = -1
        return self

    def __next__(self) -> PipelineStats:
        self._iter_idx += 1
        if self._iter_idx < len(self.pipeline_stats):
            return self.pipeline_stats[self._iter_idx]
        raise StopIteration

    def __contains__(self, item: str) -> bool:
        if not isinstance(item, str):
            item = str(item)
        res = False
        for pipeline_stats in self.pipeline_stats:
            if pipeline_stats.pipeline == item:
                res = True
                break
        return res

    def __getitem__(self, key: str) -> PipelineStats:
        return self._pipelines_dict[key]

    def __len__(self) -> int:
        return len(self.pipeline_stats)

    #
    # Internal methods
    #
    def _read_playlist_file(self) -> List[str]:
        """Read contents of playlist file, if any

        Returns:
            List[str]: contents of playlist file, a list of pipelines
        """
        if not Path.exists(self._playlist_path):
            self.logger.debug(f"{self._playlist_path} not found")
            return []

        with open(self._playlist_path, "r", encoding="utf-8") as file:
            playlist = yaml.safe_load(file)

        return playlist["playlist"]

    #
    # External methods
    #
    def add_pipeline(self, pipeline_path: Union[Path, str]) -> None:
        """Add pipeline yaml file to playlist.
           Do nothing if pipeline is already in playlist.

        Args:
            pipeline_path (Union[Path, str]): path of yaml file to add
        """
        pipeline_str = str(pipeline_path)
        if pipeline_str in self:
            self.logger.info(f"{pipeline_str} already in playlist")
            return
        pipeline_stats = PipelineStats(pipeline_str)
        self.pipeline_stats.append(pipeline_stats)
        self._pipelines_dict[pipeline_str] = pipeline_stats

    def delete_pipeline(self, pipeline_path: Union[Path, str]) -> None:
        """Delete pipeline yaml file from playlist.
           Do nothing if pipeline is not in playlist.

        Args:
            pipeline_path (Union[Path, str]): path of yaml file to delete
        """
        pipeline_str = str(pipeline_path)
        if pipeline_str in self:
            pipeline_stats = PipelineStats(pipeline_str)
            self.pipeline_stats.remove(pipeline_stats)
            self._pipelines_dict.pop(pipeline_str)

    def load_playlist_file(self) -> None:
        """Load playlist file"""
        pipelines = self._read_playlist_file()
        self.pipeline_stats: List[PipelineStats] = []
        self._pipelines_dict: Dict[str, PipelineStats] = {}
        for pipeline in pipelines:
            self.add_pipeline(pipeline)

    def save_playlist_file(self) -> None:
        """Save playlist file"""
        # construct playlist contents with full pathnames
        playlist = [str(stats) for stats in self.pipeline_stats]
        playlist_dict = {"playlist": playlist}
        self.logger.debug(f"playlist_dict={playlist_dict}")

        with open(self._playlist_path, "w", encoding="utf8") as file:
            yaml.dump(playlist_dict, file)
