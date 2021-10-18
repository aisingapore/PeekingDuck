#!/bin/sh

if ! (mypy --show-error-codes \
           --ignore-missing-imports \
           --disallow-untyped-defs \
           --disable-error-code override \
           --disable-error-code attr-defined peekingduck); then
    echo "TYPE CHECK FAIL"
    exit 123
else
    echo "TYPE CHECK SUCCESS!!"
fi