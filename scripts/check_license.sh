#!/bin/bash

dir=peekingduck
copyright="Copyright 2021 AI Singapore"
license='www.apache.org/licenses/LICENSE-2.0'
fail=false

find $dir -type f -name '*.py' | while read file; do
    if ! head -n 10 $file | grep -qE "$copyright|$license";
    then 
        echo "$file: Does not have a license header."
        fail=true
    fi
done

if $fail;
then
    echo "License checking failed, as some .py files do not have license header."
    exit 123
else
    echo "License checking passed - all .py files have license headers!"
    exit 0
fi

