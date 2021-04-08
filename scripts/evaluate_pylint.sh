#!/bin/bash

export POLYAXON_NO_OP=1
#app_name=$CI_COMMIT_REF_NAME
DIR=$(dirname "$0")
PROJ_ROOT="${DIR}"/..

echo "PYLINT NOW..."

# NOTE: can add more folders to linting folders with space as separator. e.g.
# SUB_FOLDERS="${PROJ_ROOT}/xxx/ ${PROJ_ROOT}/yyy/"
SUB_FOLDERS="${PROJ_ROOT}/src"
echo "Linting folders: ${SUB_FOLDERS}"

LINT_RESULTS=$(pylint --rcfile="${PROJ_ROOT}/.pylintrc" "${SUB_FOLDERS}" 2>&1)

echo "$LINT_RESULTS"
lint_score=$(echo "$LINT_RESULTS" | grep "Your code has been rated at" | grep -Eo "[-]{0,1}[0-9]+[.][0-9]+" | head -1)
int_score=${lint_score%.*}

if [ "$int_score" -eq "10" ]
then
    post_text="$Test passed. Lint score: ${int_score}"
    echo "$post_text"
else
    echo "Lint score failed: $int_score"
    exit 1
fi

echo -e "Congratulations. pylint passed.\n"
