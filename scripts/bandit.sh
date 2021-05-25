#!/bin/sh

if ! (bandit -r -ll .); then
    echo "BANDIT ERROR: found >= medium security vulnerability"
    exit 123
else
    echo "BANDIT SUCCESS!!"
fi

echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"