#!/bin/sh

if ! (mypy --ignore-missing-imports --strict --pretty --disable-error-code assignment  peekingduck); then
    echo "TYPE CHECK FAIL"
    exit 123
else
    echo "TYPE CHECK SUCCESS!!"
fi