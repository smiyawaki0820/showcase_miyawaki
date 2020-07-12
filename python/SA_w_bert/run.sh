#!usr/bin/bash

USAGE="Usage: bash train.sh -e [EXP_NAME] -g [GPU_ID] -i [DATA] -o [OUT_DIR] -l [LOG_LEVEL]"

while getopts e:g:i:o:l: OPT
do
  case ${OPT} in
    "e" ) FLG_E="TRUE"; EXP_NAME=${OPTARG};;
    "g" ) FLG_G="TRUE"; GPU_ID=${OPTARG};;
    "i" ) FLG_I="TRUE"; DATA=${OPTARG};;
    "o" ) FLG_O="TRUE"; OUT_DIR=${OPTARG};;
    "l" ) FLG_L="TRUE"; LOG_LEVEL=${OPTARG};;
    * ) echo ${USAGE} 1>&2
      exit 1 ;;
  esac
done

test "${FLG_E}" != "TRUE" && EXP_NAME=sample
test "${FLG_G}" != "TRUE" && GPU_ID=0
test "${FLG_I}" != "TRUE" && SRC_DATA=/work01/riki-fujihara/project-taiwa-sa/data_under_sampled.txt
test "${FLG_O}" != "TRUE" && OUT_DIR="result"
test "${FLG_L}" != "TRUE" && LOG_LEVEL=10

mkdir -p $OUT_DIR/datasets $OUT_DIR/logs $OUT_DIR/runs $OUT_DIR/images $OUT_DIR/models

DATA=$OUT_DIR/datasets/data
cp $SRC_DATA $DATA.src

### data 処理 ###
cat $DATA.src | awk '{print 1+$1}' > $DATA.label
cat $DATA.src | cut -f1 --complement > $DATA.text
paste -d \\t $DATA.label $DATA.text > $DATA.csv
# shuf $DATA -n 2   # 表示

### sed emoji ### 
sed -ie 's/😁/1/g' $DATA.csv
sed -ie 's/😠/2/g' $DATA.csv
sed -ie 's/😢/3/g' $DATA.csv
sed -ie 's/😊/4/g' $DATA.csv
sed -ie "s/\r//g" $DATA.csv   # 改行削除

export LOG_LEVEL=$LOG_LEVEL # 10:debug, 20:info, 30:warning

echo "### python src/mains/train.py ###"

for lr in 2e-5 5e-5 1e-4 ; do
  CUDA_VISIBLE_DEVICES=${GPU_ID} python src/mains/train.py \
    -exp $EXP_NAME \
    --data_path $DATA.csv \
    --f_train $OUT_DIR/datasets/train.csv \
    --f_dev $OUT_DIR/datasets/dev.csv \
    --f_test $OUT_DIR/datasets/test.csv \
    --dest $OUT_DIR \
    --seed 0 \
    --epoch 20 \
    --lr $lr \
    --max_token_len 64 \
    --data_size 100 \
    --batch_size 128 \
    --split_ratio 0.8 0.1 0.1 \
    --warmup 0 \
    --n_classes 4 \
    | tee result/logs/train_lr${lr}.log
done

echo "### python src/mains/plot_results.py ###"
python src/mains/plot_results.py \
  --fi ${OUT_DIR}/results.csv

echo "### DONE run.sh ###"
