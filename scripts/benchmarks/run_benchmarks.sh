#!/bin/bash

########################################
#                                      #
#  :                                :  #
#                                      #
#    PeekingDuck Benchmarking Suite    #
#          by dotw 2022-01-07          #
#                                      #
#  :                                :  #
#                                      #
########################################

# Check we are in the right O/S environment
case "$OSTYPE" in
    darwin*) OS="MacOS" ;;
    linux*) OS="Linux" ;;
    *)
        echo "Unsupported operating system: ${OSTYPE}"
        echo "Please run this script in Linux or MacOS, or in Windows WSL"
        exit 1
        ;;
esac

####################
#
# Global Vars
#
####################
# Benchmark Data Sources 
BM_DATA_DIR="data/benchmark"
GCP_DATA_URL="https://storage.googleapis.com/peekingduck/videos"
MULTI_PEOPLE_FILE="multiple_people.mp4"
SINGLE_PERSON_FILE="single_person.mp4"
MULTI_PEOPLE_DIR="${BM_DATA_DIR}/multi"
MULTI_PEOPLE_PATH="${MULTI_PEOPLE_DIR}/${MULTI_PEOPLE_FILE}"
MULTI_PEOPLE_URL="${GCP_DATA_URL}/${MULTI_PEOPLE_FILE}"
SINGLE_PERSON_DIR="${BM_DATA_DIR}/single"
SINGLE_PERSON_PATH="${SINGLE_PERSON_DIR}/${SINGLE_PERSON_FILE}"
SINGLE_PERSON_URL="${GCP_DATA_URL}/${SINGLE_PERSON_FILE}"
# Object Detection Models
YXLM=scripts/benchmarks/run_yolox_large_multi.yml
YXLS=scripts/benchmarks/run_yolox_large_single.yml
YXMM=scripts/benchmarks/run_yolox_medium_multi.yml
YXMS=scripts/benchmarks/run_yolox_medium_single.yml
YXSM=scripts/benchmarks/run_yolox_small_multi.yml
YXSS=scripts/benchmarks/run_yolox_small_single.yml
YXTM=scripts/benchmarks/run_yolox_tiny_multi.yml
YXTS=scripts/benchmarks/run_yolox_tiny_single.yml
# Pose Estimation Models
MPL=scripts/benchmarks/run_multipose_lightning.yml
SPL=scripts/benchmarks/run_singlepose_lightning.yml
SPT=scripts/benchmarks/run_singlepose_thunder.yml
# Working Files
LOG=/tmp/benchmark_log.txt
CURR_RUN=/tmp/benchmark_curr_run.txt
SUM_FPS=/tmp/benchmark_sum_fps.txt
SUM_SETUP=/tmp/benchmark_sum_setup.txt      # pipeline setup
SUM_STARTUP=/tmp/benchmark_sum_startup.txt  # system startup

####################
#
# Global Functions
#
####################
check_curl_wget() {
    # Checks if either curl or wget is installed
    if command -v curl &> /dev/null; then
        echo "download_curl"
    elif command -v wget &> /dev/null; then
        echo "download_wget"
    else
        echo ""
    fi
}

download_curl() {
    # This function downloads url to output_path using curl
    url=$1
    output_path=$2
    echo "curl: ${url} to ${output_path}"
    curl -o ${output_path} ${url}
}

download_wget() {
    # This function downloads url to output_path using wget
    url=$1
    output_path=$2
    echo "wget: ${url} to ${output_path}"
    wget -O ${output_path} --show-progress ${url}
}

verify_data_files() {
    # Checks that necessary benchmark data assets are present, else download them
    # Also creates the necessary data directories
    [ -d ${MULTI_PEOPLE_DIR} ] || { echo "creating ${MULTI_PEOPLE_DIR}"; mkdir -p ${MULTI_PEOPLE_DIR}; }
    [ -d ${SINGLE_PERSON_DIR} ] || { echo "creating ${SINGLE_PERSON_DIR}"; mkdir -p ${SINGLE_PERSON_DIR}; }
    [ -f ${MULTI_PEOPLE_PATH} ] || {
        echo "${MULTI_PEOPLE_PATH} not found"
        ${DOWNLOAD} ${MULTI_PEOPLE_URL} ${MULTI_PEOPLE_PATH}
    }
    [ -f ${SINGLE_PERSON_PATH} ] || {
        echo "${SINGLE_PERSON_PATH} not found"
        ${DOWNLOAD} ${SINGLE_PERSON_URL} ${SINGLE_PERSON_PATH}
    }
}

####################
#
# Startup Configurations
#
####################

# CMDS var stores the complete list of benchmarks to run
declare -a CMDS=( "${YXTS}" "${YXSS}" "${YXMS}" "${YXLS}" "${SPL}" "${SPT}" \
                    "${YXTM}" "${YXSM}" "${YXMM}" "${YXLM}" "${MPL}" )
NUM_RUNS=5      # set this to number of consecutive runs desired

# Keep this single task, single run for debugging/testing script changes
#declare -a CMDS=( "${SPL}" )
#NUM_RUNS=1

# Check we are in PeekingDuck's root folder
if [[ `pwd` == *PeekingDuck ]]; then
    echo "PeekingDuck Benchmarking"
else
    echo "Please run this script from PeekingDuck root folder, via"
    echo "> PeekingDuck$ scripts/benchmarks/run_benchmarks.sh"
    exit 1
fi

# Check we have the necessary tools/utilities
DOWNLOAD=$(check_curl_wget)
[ -z ${DOWNLOAD} ] && { echo "curl/wget not found, abort"; exit 1; }

# Ensure we have the necessary benchmarking data files
verify_data_files


####################
#
# Main Program Loop
#
####################
for cmd in ${CMDS[@]}; do
    CMD="python __main__.py --log_level debug --config_path ${cmd}"
    echo "cmd = ${CMD}"
    echo "Benchmarking over ${NUM_RUNS} consecutive runs..."

    # local vars for calculations
    sum_startup_time=0
    sum_setup_time=0
    sum_fps=0

    # Main experiment loop
    for ((i = 0; i < ${NUM_RUNS}; i++)); do
        # dotw: communicate via tmp files since cannot display output in this loop
        echo ${i} > ${CURR_RUN}

        { time -p ${CMD} &>${LOG}; } 2>&1  # do not ignore command output

        # compute cumulative startup time (NB: double cut to trim whitespace)
        startup_time=`grep "Startup time" ${LOG} | cut -d'=' -f2 | cut -d' ' -f2`
        sum_startup_time=$(echo "scale=2; ${sum_startup_time} + ${startup_time}" | bc)
        echo ${sum_startup_time} > ${SUM_STARTUP}

        # compute cumulative setup time (normally incurred by model nodes)
        setup_time=`grep "setup time" ${LOG} | grep model | cut -d'=' -f2 | cut -d' ' -f2`
        sum_setup_time=$(echo "scale=2; ${sum_setup_time} + ${setup_time}" | bc)
        echo ${sum_setup_time} > ${SUM_SETUP}

        # compute cumulative FPS
        num_fps=`grep "all proc" ${LOG} | cut -d':' -f5 | cut -d' ' -f2`
        sum_fps=$(echo "scale=2; ${sum_fps} + ${num_fps}" | bc)
        echo ${sum_fps} > ${SUM_FPS}
    done | awk '
        /real/ { real = real + $2; nr++ }
        /user/ { user = user + $2; nu++ }
        /sys/  { sys  = sys  + $2; ns++}
        END    {
                 if (nr>0) printf("Avg real time per run = %.2f sec\n", real/nr);
                 # disable the rest, but keep here in case need them in future
                 #if (nr>0) printf("real %f\n", real/nr);
                 #if (nu>0) printf("user %f\n", user/nu);
                 #if (ns>0) printf("sys %f\n",  sys/ns)
               }'

    # print average total start time (= avg startup time + avg setup time)
    sum_startup_time=`cat ${SUM_STARTUP}`
    avg_startup_time=$(echo "scale=2; ${sum_startup_time} / ${NUM_RUNS}" | bc)
    echo "Avg startup time = ${avg_startup_time} sec"

    sum_setup_time=`cat ${SUM_SETUP}`
    avg_setup_time=$(echo "scale=2; ${sum_setup_time} / ${NUM_RUNS}" | bc)
    echo "Avg setup time = ${avg_setup_time} sec"

    avg_total_start_time=$(echo "scale=2; ${avg_startup_time} + ${avg_setup_time}" | bc)
    echo "Avg total start time = ${avg_total_start_time} sec"

    # print average FPS
    sum_fps=`cat ${SUM_FPS}`
    avg_fps=$(echo "scale=2; ${sum_fps} / ${NUM_RUNS}" | bc)
    echo "Avg FPS = ${avg_fps}"
done

exit 0

