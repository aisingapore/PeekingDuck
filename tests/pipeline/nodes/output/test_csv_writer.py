# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import datetime
import re
from pathlib import Path

import pytest

from peekingduck.pipeline.nodes.output.csv_writer import Node


def directory_contents():
    return list(Path.cwd().iterdir())


@pytest.fixture
def writer():  # logging interval of 1 second
    # absolute file_path is used for the temp dir created for the test
    csv_writer = Node(
        {
            "input": "all",
            "output": "end",
            "file_path": str(Path.cwd() / "test1.csv"),
            "stats_to_track": ["bbox", "bbox_labels"],
            "logging_interval": "1",
        }
    )
    return csv_writer


@pytest.fixture
def writer2():  # logging interval of 5 second
    # absolute file_path is used for the temp dir created for the test
    csv_writer = Node(
        {
            "input": "all",
            "output": "end",
            "file_path": str(Path.cwd() / "test2.csv"),
            "stats_to_track": ["bbox", "bbox_labels"],
            "logging_interval": "5",
        }
    )
    return csv_writer


@pytest.mark.usefixtures("tmp_dir")
class TestCSVWriter:
    def test_cwd_starts_empty(self):
        assert list(Path.cwd().iterdir()) == []

    def test_check_csv_name(self, writer):
        inputs = {
            "bbox": [[1, 2, 3, 4]],
            "bbox_labels": ["person"],
            "pipeline_end": False,
        }
        for _ in range(10):
            writer.run(inputs)  # write a few entries

        final_frame = {"bbox": None, "bbox_labels": None, "pipeline_end": True}
        writer.run(final_frame)

        # check timestamp is appended to filename
        pattern = r".*_\d{6}-\d{2}-\d{2}-\d{2}\.[a-z]{3}$"

        assert len(directory_contents()) == 1
        assert directory_contents()[0].suffix == ".csv"
        assert re.search(pattern, str(directory_contents()[0]))

    def test_check_header_in_csv(self, writer):
        inputs = {
            "bbox": [[1, 2, 3, 4]],
            "bbox_labels": ["person"],
            "pipeline_end": False,
        }
        for _ in range(10):
            writer.run(inputs)  # write a few entries

        final_frame = {"bbox": None, "bbox_labels": None, "pipeline_end": True}
        writer.run(final_frame)

        with open(directory_contents()[0], newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")
            header = reader.fieldnames
            for row in reader:  # read all row entries in the reader
                pass

        assert header == ["Time", "bbox", "bbox_labels"]

    def test_check_logging_interval(self, writer2):
        inputs = {
            "bbox": [[1, 2, 3, 5]],
            "bbox_labels": ["person"],
            "pipeline_end": False,
        }

        time_lapse = 0
        start_time = datetime.datetime.now()

        while time_lapse < 15:
            curr_time = datetime.datetime.now()
            # Run writer with input arriving in 1 sec intervals
            # total 14 secs with logging interval of 5 sec
            # should log 2 entries
            if (curr_time - start_time).seconds >= 1:
                time_lapse += 1
                start_time = curr_time
                writer2.run(inputs)

        final_frame = {"bbox": None, "bbox_labels": None, "pipeline_end": True}
        writer2.run(final_frame)

        with open(directory_contents()[0], newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")
            header = reader.fieldnames
            for row in reader:  # read all row entries in the reader
                pass

            # includes header plus total data entry
            num_lines = reader.line_num

        assert header == ["Time", "bbox", "bbox_labels"]
        assert len(directory_contents()) == 1
        assert num_lines == 3  # include header

    def test_check_invalid_stats(self, writer):
        # data pool did not include bbox_labels
        # But stats to track include bbox_labels
        inputs = {"bbox": [[1, 2, 3, 5]], "pipeline_end": False}

        for _ in range(10):
            writer.run(inputs)  # write a few entries

        final_frame = {"bbox": None, "pipeline_end": True}
        writer.run(final_frame)

        with open(directory_contents()[0], newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")
            header = reader.fieldnames
            for row in reader:  # read all row entries in the reader
                pass

        assert header == ["Time", "bbox"]
