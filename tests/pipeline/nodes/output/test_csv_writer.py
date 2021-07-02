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
import os
import re

import pytest
from peekingduck.pipeline.nodes.output.csv_writer import Node

def directory_contents():
    cwd = os.getcwd()
    res = os.listdir(cwd)
    return res

@pytest.fixture
def writer():  
    csv_writer = Node({
        "input": "all",
        "output": "end",
        "filepath": os.path.join(os.getcwd(),"test.csv"), 
        "stats_to_track":["bbox","bbox_labels"],
        "logging_interval":"2"
    }) #use the absolute filepath in the temp dir created for the test
    return csv_writer

@pytest.mark.usefixtures("tmp_dir")
class TestCSVWriter:

    def test_cwd_starts_empty(self):
        assert os.listdir(os.getcwd()) == []

    def test_check_csv_name(self,writer):
        inputs={
            "bbox":[[1,2,3,4]],
            "bbox_labels":["person"]
        }
        writer.run(inputs)

        #check timestamp is appended to filename
        pattern = r".*_\d{6}-\d{2}-\d{2}-\d{2}\.[a-z]{3}$"

        assert len(directory_contents()) == 1
        assert directory_contents()[0].split(".")[-1] == "csv"
        assert re.search(pattern,directory_contents()[0])

    def test_check_header_in_csv(self,writer):
        inputs={
            "bbox":[[1,2,3,4]],
            "bbox_labels":["person"]
        }
        writer.run(inputs)

        with open(directory_contents()[0], newline="") as csvfile:
            reader=csv.DictReader(csvfile, delimiter=",")
            header = reader.fieldnames

        assert header == ["Time","bbox","bbox_labels"]

    
