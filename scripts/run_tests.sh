#!/bin/bash
# author: Jin Jun
# date: 05 Jul 2021
#
# This shellscript controls the behavior of running/printing coverage 
# and the varying extent to run the test.

readonly TEST_DIR=$PWD/peekingduck
readonly ALLOWED_EXT=(all unit mlmodel module)

selectedTest="$1"

show_coverage() {
    # shows coverage if parameter is set to true and echos information.
    # showCoverage: type(bool) - shows coverage if set to true

    showCoverage=$1
    if [[ $showCoverage = true ]]; then
        echo "Coverage report printed." 
        coverage report
        echo "To hide the coverage report.. Set showCoverage to false"
    else
        echo "Coverage report not printed. To show the report set showCoverage to true"
    fi
}

run_tests_batch() {
    # runs pytest together with coverage, allows multiple configurations
    # showCoverage: type(bool) - whether to show coverage
    # testType: type(str) - type of test suite to run

    testType=$1
    showCoverage=$2

    if ! (coverage run --source="${TEST_DIR}" -m pytest -m "${testType}"); then
        show_coverage "${showCoverage}"
        echo "TEST FAILED."
        exit 1
    else
        show_coverage "${showCoverage}"
        echo "TEST PASSED!"
    fi

    exit 0
}

run_tests_iteratively() {
    # Iteratively run a fresh pytest instance for each test script marked as
    # mlmodel due to memory constraints. Then runs the rest of the tests in two
    # batches: "not mlmodel and not module" and "module".
    #
    # showCoverage: type(bool) - whether to show coverage.
    #
    # References:
    # Convert multiline string to array: https://stackoverflow.com/a/45565601
    # Extract string before a colon: https://stackoverflow.com/a/20348214

    showCoverage=$1

    local IFS=$'\n'
    local testSuites=($(pytest -p no:warnings -qq -m mlmodel --co))
    # Prepend "-m" for pytest to run marker
    testSuites+=("-m not mlmodel and not module" "-m module")

    coverage erase
    local testSuite
    for testSuite in "${testSuites[@]}"; do
        coverage run --source="${TEST_DIR}" -a -m pytest "${testSuite%:*}"
        if [[ $? -ne 0 ]]; then
            show_coverage "${showCoverage}"
            echo "TEST FAILED."
            exit 1
        fi
    done
    show_coverage "${showCoverage}"
    echo "TEST PASSED!"
    exit 0
}

echo "Running ${selectedTest} tests in ${TEST_DIR}"

case "${selectedTest}" in 
    "all")
        run_tests_batch "" true
        ;;
    "mlmodel")
        run_tests_batch "mlmodel" false
        ;;
    "module")
        run_tests_batch "module" false
        ;;
    "post_merge")
        run_tests_iteratively true
        ;;
    "unit")
        run_tests_batch "not mlmodel and not module" false
        ;;
    *)
        echo "'${selectedTest}' is an illegal argument, choose from: ${ALLOWED_EXT[@]}"
        exit 1
        ;;
esac
