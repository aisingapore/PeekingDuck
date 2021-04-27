#!/bin/sh

if ! (mypy peekingduck); then
    echo "MYPY ERROR"
    exit 123
else
    echo "MYPY SUCCESS!!"
fi

echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"