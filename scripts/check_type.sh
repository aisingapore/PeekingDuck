#!/bin/sh

if ! (mypy --ignore-missing-imports peekingduck); then
    echo "TYPE CHECK FAIL"
    exit 123
else
    echo "TYPE CHECK SUCCESS!!"
fi