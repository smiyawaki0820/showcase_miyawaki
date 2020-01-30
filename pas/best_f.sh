#!/usr/bin/env bash

DIRS=("yans" "nlp2019_label_emb" "nlp2019_label_emb_soft" "nlp2019_top_layer" "nlp2019_gate")

for DIR in ${DIRS[@]}
do
echo -e "\e[31m${DIR}\e[0m"
FILES=($(ls ${DIR}/result/edit/log/*.csv))
for FILE in ${FILES[@]}
do
echo ${FILE}
python src/best_f.py \
--f ${FILE}
done
done
