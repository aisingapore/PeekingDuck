# Copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Record the nodes outputs to csv file
"""

from datetime import datetime
import logging
import textwrap
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.output.utils.csvlogger import CSVLogger


class Node(AbstractNode):
    """Node that tracks user-specified parameters and outputs the results in a
    CSV file.

    Inputs:
        ``All`` (:obj:`List`): A placeholder that represents a flexible input.
        Actual inputs to be written into the csv can be configured in
        `stats_to_track`.

    Outputs:
        None

    Configs:
        stats_to_track (:obj:`List`): **default = ["keypoints", "bboxes", "bbox_labels"]**

            Parameters to log into the CSV file. The chosen parameters must be
            present in the data pool.

        filepath (:obj:`str`): **default = "PeekingDuck/data/stats.csv"**

            Directory where CSV file is saved.

        logging_interval (:obj:`int`): **default = 1**

            Interval between each log, in terms of seconds.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.logging_interval = int(self.logging_interval) # type: ignore
        # check if filepath has a '.csv' extension
        if not ".csv" in self.filepath:
            raise ValueError("Filepath must have a '.csv' extension.")

        self._filepath_datetime = self._append_datetime_filepath(self.filepath)
        self._stats_checked = False
        self.stats_to_track = self.stats_to_track  # type: ignore
        self.csv_logger = CSVLogger(self._filepath_datetime,
                                    self.stats_to_track,
                                    self.logging_interval)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write the current state of the tracked statistics into
        the csv file as a row entry

        Args:
            inputs(dict): the data pool of the pipeline

        Returns:
            outputs: [None]
        """

        # reset and terminate when there are no more data
        if inputs["pipeline_end"]:
            self._reset()
            return {}

        if not self._stats_checked:
            self._check_tracked_stats(inputs)
            # self._stats_to_track might change after the check
            self.csv_logger = CSVLogger(self._filepath_datetime,
                                        self.stats_to_track,
                                        self.logging_interval)

        self.csv_logger.write(inputs, self.stats_to_track)

        return {}

    def _check_tracked_stats(self, inputs: Dict[str, Any]) -> None:
        """
        Check whether user input statistics is present in the data pool
        of the pipeline. Statistics not present in data pool will be
        ignored and dropped.
        """
        valid = []
        invalid = []

        for stat in self.stats_to_track:
            if stat in inputs:
                valid.append(stat)
            else:
                invalid.append(stat)

        if len(invalid) != 0:
            msg = textwrap.dedent(f"""\
                    {invalid} are not valid outputs.
                    Data pool only has this outputs: {list(inputs.keys())}
                    Only {valid} will be logged in the csv file""")
            self.logger.warning(msg)

        # update stats_to_track with valid stats found in data pool
        self.stats_to_track = valid
        self._stats_checked = True

    def _reset(self) -> None:
        del self.csv_logger

        # initialize for use in run
        self._stats_checked = False

    @staticmethod
    def _append_datetime_filepath(filepath: str) -> str:
        """
        Append time stamp to the filename
        """
        current_time = datetime.now()  # type: ignore
        time_str = current_time.strftime(
            "%d%m%y-%H-%M-%S")  # output as '240621-15-09-13'

        file_name = filepath.split('.')[-2]
        file_ext = filepath.split('.')[-1]

        # append timestamp to filename before extension
        # Format: filename_timestamp.extension
        filepath_with_timestamp = f"{file_name}_{time_str}.{file_ext}"

        return filepath_with_timestamp
