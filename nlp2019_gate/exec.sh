#!/usr/bin/env bash

USAGE="Usage: bash train.sh -g [GPU_ID] -i [IN_DIR] -o [OUT_DIR]"

### parameter ###
INFORM=false
lr=0.0001
depth=6
size=60

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

### mkdir ###
DIR=("work/" "src/")
for dir in ${DIR[@]}
do
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir"
  fi
done

### python ###
for i in 8
do

### logging date ###
read s Y m d H M S ms ns <<< "$(date + '%s %Y %m %d %H ')"
echo $Y.$m.$d $H:$M:$S >> gate.log

### train ###
TH=`echo "scale=2; ${i} / 10.0" | bc`
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python src/train_edit.py \
--data ${IN_DIR} \
--out_dir ${OUT_DIR} \
--model e2e-stack \
--vec_u 256 \
--depth ${depth} \
--optimizer adam \
--lr ${lr} \
--dropout-u 0.1 \
--size ${size} \
--model_no ${m}${d} \
--epoch 100 \
--iter 3 \
--threshold ${TH} \
| tee -a work/gate.log

MODEL_ID=$(cat work/model_id.txt)

### test ###
THRES="0.5 0.5 0.5"
MODEL_FILE="${OUT_DIR}/edit/model-${MODEL_ID}.h5"

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/test_e2e_arg_model_output_json.py \
--data ${IN_DIR} \
--tag test \
--model e2e-stack \
--model-file ${MODEL_FILE} \
--thres ${THRES} \
--vec_u 256 \
--depth ${depth} \
--optimizer adam \
--lr ${lr} \
--dropout-u 0.1 \
--model_no ${m}${d} \
--iter 3 \
--threshold ${TH} \
| tee -a work/gate.log

### inform ###
if ${INFORM}; then
  curl -X POST -H 'Content-type: application/json' --data '{"text":"finish layer iteration"}' https://hooks.slack.com/services/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
fi

done
