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
# rm ./work/hoge.txt

for lr in 0.0001
do
for i in 8
do
TH=`echo "scale=2; ${i} / 10.0" | bc`
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python src/train_edit.py \
--data ${IN_DIR} \
--out_dir ${OUT_DIR} \
--model e2e-stack \
--vec_u 256 \
--depth 6 \
--optimizer adam \
--lr ${lr} \
--dropout-u 0.1 \
--size 1 \
--model_no 0 \
--epoch 1 \
--iter 2 \
--cache "False" \
--threshold ${TH}
done
done
if ${INFORM}; then
  curl -X POST -H 'Content-type: application/json' --data '{"text":"finish yans"}' https://hooks.slack.com/services/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
fi

