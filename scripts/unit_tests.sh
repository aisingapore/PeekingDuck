#!/bin/bash

test_dir=$PWD/peekingduck
echo "Testing code in: $test_dir"

if ! (coverage run --source="$test_dir" -m pytest -m "not mlmodel"); then
    coverage report
    echo "UNIT TESTING FAILED."
    exit 123
else
    coverage report
    echo "UNIT TESTING PASSED!"
fi