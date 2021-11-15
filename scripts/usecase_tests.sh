#!/bin/bash
# author: jin jun
# date  : 31/05/2021
# this script runs all usecases and throws an error if there are crashes
# Attempts a re-run with exit code 3 (optional package installed)

for use_case in "tests/use_cases"/*
do
    command="peekingduck run --config_path='$use_case'"
    eval $command
    ret_code=$?

    if [ $ret_code -eq 3 ]; then
        eval $command
        ret_code=$?
    fi
    if [ $ret_code -ne 0 ]; then
        echo "USECASE TESTING $use_case FAILED."
        exit 123
    fi
done

echo "SYSTEM TESTING PASSED!"