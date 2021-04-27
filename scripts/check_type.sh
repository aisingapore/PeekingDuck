#!/bin/sh

if ! (mypy peekingduck); then
    echo "TYPE CHECK FAIL"
    exit 123
else
    echo "TYPE CHECK SUCCESS!!"
fi