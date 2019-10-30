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

INFORM=true


#for ((i=0; i<=10; i++))
#do
#TH=`echo "scale=2; ${i} / 10.0" | bc`
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python src/train_edit.py \
--data ${IN_DIR} \
--out_dir ${OUT_DIR} \
--model e2e-stack \
--vec_u 256 \
--depth 6 \
--optimizer adam \
--lr 0.0001 \
--dropout-u 0.1 \
--size 60 \
--model_no 1028 \
--epoch 100 \
--iter 3 \
--threshold 0.8
#done

if ${INFORM}; then
  curl -X POST -H 'Content-type: application/json' --data '{"text":"label emb null"}' https://hooks.slack.com/services/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
fi

