
USAGE="Usage: bash test.sh -g [GPU_ID] -i [IN_DIR] -o [OUT_DIR]"

#LRate=(0.0001 0.0002 0.0005 0.001)
LRate=0.0002
thresholds=(0.8)


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


for th in ${thresholds[@]}
do
for lr in ${LRate[@]}
do

# model-e2e-stack_ve256_vu256_depth6_adam_lr0.0002_du0.1_dh0.0_True_size60_sub1118_th0.8_it3_rs2020_preFalse_cacheFalse.h5
for FILE in result/edit/*depth6*lr${lr}*size60*sub1118*th${th}*cacheTrue.h5
do
if [ -f ${FILE} ] ; then
echo $FILE
THRES="0.5 0.5 0.5"

for ITER in 1 2 3
do
CUDA_VISIBLE_DEVICES=${GPU_ID} python src/test_e2e_arg_model_output_json.py \
--data ${IN_DIR} \
--tag "dev" \
--model e2e-stack \
--model-file ${FILE} \
--thres ${THRES} \
--vec_u 256 \
--depth 6 \
--optimizer adam \
--lr ${lr} \
--dropout-u 0.1 \
--model_no 1118 \
--iter ${ITER} \
--cache "False" \
--threshold ${th}
done
fi
done
done
done

curl -X POST -H 'Content-type: application/json' --data '{"text":"yans_test_finish"}' https://hooks.slack.com/services/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


