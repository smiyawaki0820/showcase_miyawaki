#!usr/bin/bash -ev

set -e

#!/usr/bin/env bash

USAGE="Usage: bash train.sh -g [GPU_ID] -e [ESTIMATE]"

while getopts g:e: OPT
do
  case ${OPT} in
    "g" ) FLG_G="TRUE"; GPU_ID=${OPTARG};;
    "e" ) FLG_E="TRUE"; ESTIMATE=${OPTARG};;
    * ) echo ${USAGE} 1>&2
      exit 1 ;;
  esac
done

<< COMMENT
# run.sh
hoge.py
hoge.py
hoge.py
COMMENT

gpu_available() {
  FREE_PER=$(( 100 * $2 / $1 ))
  #echo "free % ::: $FREE_PER"
  
  # estimate がない場合 -> 80% free だったら使用
  if [ -z ${ESTIMATE} ] ; then
    [ $FREE_PER -ge 80 ] && AVAILABLE=true
  else
    [ $(( $2 - $ESTIMATE )) -ge 1 ] && AVAILABLE=true
  fi

  echo $AVAILABLE
}

# nvidia-smi dmon -h
# nvidia-smi dmon -s u -c 1

temp_file=$(mktemp)
temp_command=$(mktemp)
temp_dir=$(mktemp -d)

trap "
rm -f $temp_file $temp_command
rm -rf $temp_dir
" 0

python nvidia-smi.py -fo $temp_file
# echo "CREATE $temp_file"

USING_GPU=()
while IFS= read -r line
do
  GPU_ID=`echo $line | jq -r '.index'`
  TOTAL=`echo $line | jq -r '.memory_total'`
  FREE=`echo $line | jq -r '.memory_free'`
  USED=`echo $line | jq -r '.memory_used'`

  isAVAILABLE=`gpu_available $TOTAL $FREE $USED`
  if [ $isAVAILABLE ] ; then
    USING_GPU+=( "${GPU_ID}" )
  fi

done < ${temp_file}

#nvidia-smi -q -d MEMORY
SW_TOTAL=`free -m | grep 'Swap:' | awk '{print $2}'`
SW_USED=`free -m | grep 'Swap:' | awk '{print $3}'`
CPU_SWAP_RATE=$(( 100 * $SW_USED / $SW_TOTAL ))

if [ $CPU_SWAP_RATE -ge 40 ] ; then
  # echo "swap rate ::: $CPU_SWAP_RATE %"
  exit
fi

ALLOCATE=()
COUNT=0
N_GPU=`echo ${#USING_GPU[@]}`
while IFS= read -r process
do
  echo -e "${USING_GPU[$(( $COUNT % $N_GPU ))]}\t$process" >> $temp_command
  COUNT=$(( $COUNT + 1 ))
done < run.txt

cat $temp_command | parallel --colsep '\t' -a - --jobs 85% --load 50% --memfree --noswap \
  'CUDA_VISIBLE_DEVICES={1} python {2}'


# -a <file> ::: ファイルの各行を引数としてコマンドを並列に実行
