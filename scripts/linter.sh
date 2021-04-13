#!/bin/sh

MINLINTSCORE=10

if ! (pylint --fail-under=$MINLINTSCORE peekingduck); then
    echo "PYLINT ERROR: score below required lint score"
    exit 123
else
    echo "PYLINT SUCCESS!!"
fi


# run pytest
pytest

echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"
