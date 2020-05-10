#!/usr/bin/env bash

USAGE="Usage: bash train.sh -g [GPU_ID] -i [IN_DIR] -o [OUT_DIR]"

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

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train_base.py \
--data ${IN_DIR} \
--out_dir ${OUT_DIR} \
--model e2e-stack \
--size 60 \
--vec_u 256 \
--optimizer adam \
--lr 0.0002 \
--dropout-u 0.1 \
--model_no 1118 \
--depth 6 \
--epoch 100

