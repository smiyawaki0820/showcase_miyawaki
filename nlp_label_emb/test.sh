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

#cp ./result/edit/hoge/e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub1_* ${OUT_DIR}/hoge/
#
#cp ./result/edit/log/model-e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub1_* ${OUT_DIR}/log/

#cp ./result_new/edit/model-* ${OUT_DIR}/


#MODEL_FILE="~/Club-IMI-taiwa-2019/out_epoch/result_new/edit/model-e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub0_th0.8_it3_rs2018_preFalse.h5"
THRES="0.5 0.5 0.5"


#for i in 0 1 2 5 10
#do
#
#TH=`echo "scale=1; ${i} / 10.0" | bc`
#
#if [ ${TH} = 0 ]; then
#  MODEL_FILE="/model-e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub1_th0.${TH}_it3_rs2016_preFalse.h5"
#  MODEL_result="Club-IMI-taiwa-2019/out_epoch/result/edit/predict-test-model-e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub1_th0.${TH}_it3_rs2016_preFalse-0.5-0.5-0.5.txt"
#elif [ ${TH} = 1.0 ]; then
#  MODEL_FILE="/model-e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub1_th${TH}_it3_rs2016_preFalse.h5"
#  MODEL_result="Club-IMI-taiwa-2019/out_epoch/result/edit/predict-test-model-e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub1_th${TH}_it3_rs2016_preFalse-0.5-0.5-0.5.txt"
#else
#  MODEL_FILE="/model-e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub1_th0${TH}_it3_rs2016_preFalse.h5"
#  MODEL_result="Club-IMI-taiwa-2019/out_epoch/result/edit/predict-test-model-e2e-stack_ve256_vu256_depth10_adam_lr0.0001_du0.1_dh0.0_True_size100_sub1_th0${TH}_it3_rs2016_preFalse-0.5-0.5-0.5.txt"
#fi

MODEL_FILE="model-e2e-stack_ve256_vu256_depth6_adam_lr0.0001_du0.1_dh0.0_True_size60_sub1028_th0.8_it3_rs2016_preFalse.h5"

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/test_e2e_arg_model_output_json.py \
--data ${IN_DIR} \
--tag test \
--model e2e-stack \
--model-file "result/edit/${MODEL_FILE}" \
--thres ${THRES} \
--vec_u 256 \
--depth 6 \
--optimizer adam \
--lr 0.0001 \
--dropout-u 0.1 \
--model_no 1 \
--iter 3 \
--threshold 0.8

curl -X POST -H 'Content-type: application/json' --data '{"text":"test_finish"}' https://hooks.slack.com/services/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

#cp ${MODEL_result} ${OUT_DIR}/hoge/predict/${TH}.json

#done

#cd ~/Club-IMI-taiwa-2019/out_epoch/result_new/edit/hoge
