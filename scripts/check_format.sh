#!/bin/sh -eu

fail="0"

black --version

if ! black --check .; then
    fail="1"
fi

if [ "$fail" = "1" ]; then
    echo "At least one file is poorly formatted - check the output above"
    exit 123
fi

echo "Everything seems to be formatted properly!"
