#!/bin/bash

test_dir=$PWD/peekingduck
echo "Testing ML Models in: $test_dir"

if ! (coverage run --source="$test_dir" -m pytest -m "mlmodel"); then
    coverage report
    echo "ML Model TESTS FAILED."
    exit 123
else
    coverage report
    echo "ML Model TESTS PASSED!"
fi