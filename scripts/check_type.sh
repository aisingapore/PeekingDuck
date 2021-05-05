#!/bin/sh

if ! (mypy --ignore-missing-imports --strict --disable-error-code assignment peekingduck); then
    echo "TYPE CHECK FAIL"
    exit 123
else
    echo "TYPE CHECK SUCCESS!!"
fi