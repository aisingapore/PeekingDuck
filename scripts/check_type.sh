#!/bin/sh

if ! (mypy --ignore-missing-imports \
           --disallow-untyped-defs \
           --disable-error-code override peekingduck); then
    echo "TYPE CHECK FAIL"
    exit 123
else
    echo "TYPE CHECK SUCCESS!!"
fi