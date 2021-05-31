#!/bin/bash
# author: jin jun
# date  : 31/05/2021
# this script runs all usecases and throws an error if there are crashes

for use_case in "use_cases"/*
do
    if ! (peekingduck run --config_path="$use_case"); then
        echo "USECASE TESTING $use_case FAILED."
        exit 123
    fi
done

echo "SYSTEM TESTING PASSED!"