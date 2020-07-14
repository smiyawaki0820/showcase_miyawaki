#!/usr/bin/bash
set -e

USAGE="
bash $0 \
  -g [gpu 0] \
  -i [data_dir datasets/data-bin] \
  -o [dest_dir results]
"

<<TODO
$dest_dir/
|- logs/
|- runs/
|- models/
|- runs/
|- datasets/
|  |- datasets.tar.gz
|- parameters.json

TODO


while getopts g:i:o: OPT
do
  case ${OPT} in
  g ) FLG_G="TRUE"; gpu=${OPTARG};;
  i ) FLG_I="TRUE"; data_dir=${OPTARG};;
  o ) FLG_O="TRUE"; dest_dir=${OPTARG};;
  * ) echo ${USAGE} 1>&2 ; exit 1 ;;
  esac
done

test "${FLG_G}" != "TRUE" && gpu=0
test "${FLG_I}" != "TRUE" && data_dir=datasets/data-bin
test "${FLG_O}" != "TRUE" && dest_dir=results

log_dir=$dest_dir/logs
run_dir=$dest_dir/runs
mkdir -p $log_dir $run_dir


# https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train
CUDA_VISIBLE_DEVICES=$gpu fairseq-train $data_dir \
  --arch transformer --seed 0 --max-epoch 5 \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
  --dropout 0.3 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4096 \
  --update-freq 2 \
  --save-dir $dest_dir/models \
  --log-interval 1 --tensorboard-logdir $run_dir \
  | tee $log_dir/train.log


