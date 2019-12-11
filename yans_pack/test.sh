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
    OUT_DIR="result"
fi

THRES="0.5 0.5 0.5"
depth=6
LRate=(0.002)
size=60
sub=1127
IT=(3)
echo -e "\e[32mpre\e[m"
read pre
echo -e "\e[32mloss\e[m"
read loss
save=true
save_json="True"
rs=
null=("inc" "exc")

for it in ${IT[@]}
do

if [ ${it} == 1 ] ; then
echo -e "\e[32mbase\e[m"
test_iter=(1)
else
echo -e "\e[32mno base\e[m"
test_iter=(1 2 3)
fi


for ITER in ${test_iter[@]}
do

for lr in ${LRate[@]}
do

for th in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
for n in ${null[@]}
do
for FILE in result/edit/model*new*e2e*depth${depth}*lr${lr}*size${size}*sub${sub}*th${th}*it${it}*rs${rs}*pre${pre}*loss-${loss}*null-${n}*
do
if [ -f ${FILE} ] ; then

echo -e "\e[31miter ${ITER} \e[m"
echo -e "\e[34m___ ${FILE}\e[m"

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/test_e2e_arg_model_output_json.py \
--data ${IN_DIR} \
--tag "dev" \
--null_label ${null} \
--model e2e-stack \
--model-file ${FILE} \
--thres ${THRES} \
--vec_u 256 \
--depth ${depth} \
--optimizer adam \
--lr ${lr} \
--dropout-u 0.1 \
--model_no ${sub} \
--iter ${ITER} \
--threshold ${th} \
--load ${pre} \
--save_json "True" \
--batch 512

fi
if [ $save == false ] ; then
  ls=$(result/*${FILE}*)
  echo ? - rm result/$ls
  read inpu
  if [ $inpu == "True" ] ; then
    rm result/*FILE*
  fi
fi
done
done
done
done
done
done

# curl -X POST -H 'Content-type: application/json' --data '{"text":"test_finish"}' https://hooks.slack.com/services/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

