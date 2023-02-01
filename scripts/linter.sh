#!/bin/sh


if ! (pylint peekingduck); then
    echo "PYLINT ERROR: score below required lint score"
    exit 123
else
    echo "PYLINT SUCCESS!!"
fi

echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"
