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
    OUT_DIR="result/"
fi

depth=6
lr=0.0002
sub=1118


for LR in ${lr[@]}
do
for FILE in result/base/*depth6*lr${LR}*size60*sub${sub}*.h5
do
if [ -f ${FILE} ] ; then
echo ${FILE}
THRES="0.5 0.5 0.5"

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/test_e2e_arg_model_output_json.py \
--data ${IN_DIR} \
--tag "dev" \
--model e2e-stack \
--model-file ${FILE} \
--thres ${THRES} \
--vec_u 256 \
--depth ${depth} \
--optimizer adam \
--lr ${LR} \
--dropout-u 0.1 \
--model_no ${sub}
fi
done
done

curl -X POST -H 'Content-type: application/json' --data '{"text":"test_finish"}' https://hooks.slack.com/services/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
