#!/usr/bin/bash
set -e 

USAGE="
bash $0 \
  -g [gpu 0] \
  -i [model results/models/checkpoint_last.pt]
"

while getopts g:i:o: OPT ; do
  case ${OPT} in
  g ) FLG_G="TRUE"; gpu=${OPTARG};;
  i ) FLG_I="TRUE"; model=${OPTARG};;
  * ) echo ${USAGE} 1>&2 ; exit 1 ;;
  esac
done

test "${FLG_G}" != "TRUE" && gpu=0
test "${FLG_I}" != "TRUE" && model=results/models/checkpoint_last.pt

data_dir=datasets/data-bin

CUDA_VISIBLE_DEVICES=$gpu python src/interactive-for-sa.py \
  --source-lang src --target-lang tgt \
  --path $model $data_dir \
  --nbest 5
