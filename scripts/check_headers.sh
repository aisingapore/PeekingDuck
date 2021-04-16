#!/bin/bash

dir=peekingduck
copyright='Copyright 2021 AI Singapore'
license='www.apache.org/licenses/LICENSE-2.0'
fail=false

# For easy troubleshooting, show all the files which do not have the required headers
for file in $(find $dir -type f -name '*.py'); do
    if ! head -n 10 $file | grep -q "$copyright"
    then 
        echo "$file: Does not have AISG copyright header."
        fail=true
    fi
    if ! head -n 10 $file | grep -q "$license"
    then 
        echo "$file: Does not have Apache license header."
        fail=true
    fi
done

if $fail;
then
    echo "Header checking failed, as some .py files do not have copyright or license headers."
    exit 123
else
    echo "Header checking passed - all .py files have copyright and license headers!"
    exit 0
fi

