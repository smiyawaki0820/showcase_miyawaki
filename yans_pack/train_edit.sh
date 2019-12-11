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

INFORM=false
# rm ./work/hoge.txt
loss_type="sum" # sum or last
ITER=(3)
depth=6
size=60
model_no=1127
batch_size=512
lr=(0.002)
load="False"
null="exc"
free="new"

rm result/edit/log/*sub0*

for lr in ${lr[@]}
do
for it in ${ITER[@]}
do

if [ ${it} == 1 ] ; then
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python src/train_edit.py \
--free ${free} \
--data ${IN_DIR} \
--out_dir ${OUT_DIR} \
--model e2e-stack \
--vec_u 256 \
--depth ${depth} \
--optimizer adam \
--lr ${lr} \
--dropout-u 0.1 \
--size ${size} \
--model_no ${model_no} \
--epoch 100 \
--iter 1 \
--batch ${batch_size} \
--threshold 0.0 \
--load "False" \
--null_label ${null} \
--loss ${loss_type}
else
for TH in 0.8 0.2 0.3 0.5 0.0
do
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python src/train_edit.py \
--free ${free} \
--data ${IN_DIR} \
--out_dir ${OUT_DIR} \
--model e2e-stack \
--vec_u 256 \
--depth ${depth} \
--optimizer adam \
--lr ${lr} \
--dropout-u 0.1 \
--size ${size} \
--model_no ${model_no} \
--epoch 100 \
--iter ${it} \
--batch ${batch_size} \
--threshold ${TH} \
--load ${load} \
--null_label ${null} \
--loss ${loss_type}
done
fi
done
done

if ${INFORM}; then
  curl -X POST -H 'Content-type: application/json' --data '{"text":"finish layer iteration"}' https://hooks.slack.com/services/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
fi

