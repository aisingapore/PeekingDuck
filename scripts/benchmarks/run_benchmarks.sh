#!/bin/bash

########################################
#                                      #
#   :                              :   #
#                                      #
#    PeekingDuck Benchmarking Suite    #
#          by dotw 2022-01-07          #
#                                      #
#   :                              :   #
#                                      #
########################################

####################
#
# Global vars
#
####################
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
SUM_STARTUP=/tmp/benchmark_sum_startup.txt

####################
#
# Startup configurations
#
####################

# CMDS var stores the complete list of benchmarks to run
declare -a CMDS=( "${YXTS}" "${YXSS}" "${YXMS}" "${YXLS}" "${SPL}" "${SPT}" \
                    "${YXTM}" "${YXSM}" "${YXMM}" "${YXLM}" "${MPL}" )
NUM_RUNS=5      # set this to number of consecutive runs desired

# Keep this single task, single run for debugging/testing script changes
#declare -a CMDS=( "${YXTS}" )
#NUM_RUNS=1

# Check we are in PeekingDuck/ root folder, else abort
if [[ `pwd` == *PeekingDuck ]]; then
    echo "PeekingDuck Benchmarking"
else
    echo "Please run this script in PeekingDuck root folder"
    exit 1
fi

####################
#
# Main program loop
#
####################
for cmd in ${CMDS[@]}; do
    CMD="python __main__.py --config_path ${cmd}"
    echo "cmd = ${CMD}"
    echo "Benchmarking over ${NUM_RUNS} consecutive runs..."

    sum_startup_time=0
    sum_fps=0

    # Main experiment loop
    for ((i = 0; i < ${NUM_RUNS}; i++)); do
        # dotw: communicate via tmp files since cannot display output in this loop
        echo ${i} > ${CURR_RUN}

        { time -p ${CMD} &>${LOG}; } 2>&1  # do not ignore command output

        # compute cumulative startup time
        # NB: double cut to trim whitespace
        startup_time=`grep "Startup delay" ${LOG} | cut -d'=' -f2 | cut -d' ' -f2`
        sum_startup_time=$(echo "scale=4; ${sum_startup_time} + ${startup_time}" | bc)
        echo ${sum_startup_time} > ${SUM_STARTUP}

        # compute cumulative FPS
        # NB: double cut to trim whitespace
        num_fps=`grep "all proc" ${LOG} | cut -d':' -f5 | cut -d' ' -f2`
        sum_fps=$(echo "scale=2; ${sum_fps} + ${num_fps}" | bc)
        echo ${sum_fps} > ${SUM_FPS}
    done | awk '
        /real/ { real = real + $2; nr++ }
        /user/ { user = user + $2; nu++ }
        /sys/  { sys  = sys  + $2; ns++}
        END    {
                 if (nr>0) printf("Avg real time per run = %.4f sec\n", real/nr);
                 # disable the rest, but keep here in case need them in future
                 #if (nr>0) printf("real %f\n", real/nr);
                 #if (nu>0) printf("user %f\n", user/nu);
                 #if (ns>0) printf("sys %f\n",  sys/ns)
               }'

    # print average startup delay and FPS
    sum_startup_time=`cat ${SUM_STARTUP}`
    avg_startup_time=$(echo "scale=4; ${sum_startup_time} / ${NUM_RUNS}" | bc)
    echo "Avg startup delay = ${avg_startup_time} sec"

    sum_fps=`cat ${SUM_FPS}`
    avg_fps=$(echo "scale=2; ${sum_fps} / ${NUM_RUNS}" | bc)
    echo "Avg FPS = ${avg_fps}"
done

exit 0

