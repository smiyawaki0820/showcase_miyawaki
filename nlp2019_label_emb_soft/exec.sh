#!/usr/bin/env bash

USAGE="Usage: bash train.sh -g [GPU_ID] -i [IN_DIR] -o [OUT_DIR]"

### parameter ###
TRAIN=true
TEST=true
INFORM=true
SAVE=true
LRate=(0.0001 0.0002 0.0005 0.001)
depth=6
size=60
epoch=100
model_no=1030
NULL_LABEL="inc" # inc/exc

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
for lr in ${LRate[@]}
do

### logging date ###
read s Y m d H M S ms ns <<< "$(date +'%s %Y %m %d %H %M %S %3N %9N')"
echo $Y.$m.$d $H:$M:$S >> gate.log

### train ###
if ${TRAIN}; then
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
--model_no ${model_no} \
--epoch ${epoch} \
--iter 3 \
--threshold 0.5 \
--null_label ${NULL_LABEL} \
| tee -a work/gate.log

MODEL_ID=$(cat work/model_id.txt)
MODEL_FILE="${OUT_DIR}/edit/model-${MODEL_ID}.h5"
fi


### test ###
if ${TEST}; then
THRES="0.5 0.5 0.5"

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
--model_no ${model_no} \
--iter 3 \
--threshold 0.5 \
--null_label ${NULL_LABEL} \
| tee -a work/gate.log
fi

### inform ###
if ${INFORM}; then
  lsc=$(tmux lsc)
  if [ -n "${lsc}" ]; then
    mes="finish gate"#\n tmux: ${lsc}\n ${HOST}"
  else
    mes="finish gate"#\n ${HOST}"
  fi
  echo ${HOST}
  data='{"text":"finish_gate:'"${HOST}"'"}'
  curl -X POST -H 'Content-type: application/json' --data ${data} https://hooks.slack.com/services/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
fi

done

### confirm save ###
function ConfirmSave() {

  read input

  if [ -z $input ] ; then
    echo "save model - yes/no"
    ConfirmSave

  elif [ $input = 'no' ] || [ $input = 'NO' ] || [ $input = 'n' ] ; then
    rm ${MODEL_FILE}
    rm result/edit/hoge/${MODEL_ID}_WRjudge.txt
    rm result/edit/log/model-${MODEL_ID}*
    rm result/predict-test-model-${MODEL_ID}*
    rm result/predict/test_model-${MODEL_ID}*
  fi

}

# シェルスクリプトの実行を継続するか確認します。
if ${SAVE}; then
  echo "model saved"
else
  echo -e "\e[31m save model? - y/n \e[m"
  ConfirmSave
fi

