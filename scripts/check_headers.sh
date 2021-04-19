#!/bin/bash

dir=peekingduck
copyright='AI Singapore'
license='www.apache.org/licenses/LICENSE-2.0'
top_rows=15 # assuming copyright and license are within first top_rows of file
fail=false

# For easy troubleshooting, show all the files which do not have the required headers
for file in $(find $dir -type f -name '*.py'); do
    if ! head -n $top_rows $file | grep -q "$copyright"
    then 
        echo "$file: Does not have AISG copyright header."
        fail=true
    fi
    if ! head -n $top_rows $file | grep -q "$license"
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

