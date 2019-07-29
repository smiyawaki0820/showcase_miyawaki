#!/usr/bin/env bash

USAGE="Usage: bash test.sh -g [GPU_ID] -i [IN_DIR] -o [OUT_DIR]"

while getopts g:i:o: OPT
do
    case ${OPT} in
        "g" ) FLG_G="TRUE"; GPU_ID=${OPTARG};;
        "i" ) FLG_I="TRUE"; IN_DIR=${OPTARG};;
        "o" ) FLG_O="TRUE"; OUT_DIR=${OPTARG};;
        * ) echo ${USAGE} 1>&2
            exit 1 ;;
    esac
done

if test "${FLG_I}" != "TRUE"; then
    echo ${USAGE} 1>&2
    exit 1
elif test "${FLG_O}" != "TRUE"; then
    OUT_DIR="result"
fi

#MODEL_FILE="model-e2e-stack_ve256_vu256_10_adam_lr0.0002_du0.1_dh0.0_True_size100_sub0.h5"
#LOG_FILE="log-train-e2e-stack_ve256_vu256_10_adam_lr0.0002_du0.1_dh0.0_True_size100_sub0.txt"
#THRES=`tail "${OUT_DIR}/log/${LOG_FILE}" | grep -o "best in epoch .*]" | grep -o "\[.*\]" | grep -o "[0-9.]*" | tr '\n' ' '`

MODEL_FILE="model-e2e-stack_ve256_vu256_4_adam_lr0.0002_du0.1_dh0.0_True_size10_sub0.h5"
THRES="0.5 0.5 0.5"

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/test_e2e_arg_model_output_json.py \
--data ${IN_DIR} \
--tag test \
--model e2e-stack \
--model-file "${OUT_DIR}/${MODEL_FILE}" \
--thres ${THRES} \
--vec_u 256 \
--depth 4 \
--optimizer adam \
--lr 0.0002 \
--dropout-u 0.1 \
--model_no 0