#!/bin/sh

MINLINTSCORE=10

if ! (pylint --fail-under=$MINLINTSCORE --extension-pkg-whitelist=cv2 --min-similarity-lines=12 peekingduck); then
    echo "PYLINT ERROR: score below required lint score"
    exit 123
else
    echo "PYLINT SUCCESS!!"
fi

echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"