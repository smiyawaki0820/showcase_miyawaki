#!/usr/bin/bash
set -e

USAGE="
bash sh/preprocess.sh \
  -i [data_dir datasets/parallel]
"

<<TODO
datasets/ 
|- parallel (input data)
|  |- train.src
|  |- train.tgt
|  |- dev.src
|  |- dev.tgt
|  |- test.src
|  |- test.tgt
|- data-bin
|- 

TODO


while getopts i: OPT ; do
  case ${OPT} in
  i ) FLG_I="TRUE"; data_dir=${OPTARG};;
  * ) echo ${USAGE} 1>&2 ; exit 1 ;;
  esac
done

test "${FLG_I}" != "TRUE" && data_dir=datasets/parallel


# Preprocess =================

dest_dir=datasets/data-bin
log_dir=datasets/logs
mkdir -p $dest_dir $log_dir

fairseq-preprocess \
  --source-lang src --target-lang tgt \
  --trainpref $data_dir/train \
  --validpref $data_dir/dev \
  --testpref $data_dir/test \
  --destdir datasets/data-bin \
  --workers 6 \
  | tee $log_dir/preprocess.log

