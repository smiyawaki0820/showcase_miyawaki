## AR PAS
https://scrapbox.io/labskowmywk-61304541/PAS-pytorch
https://scrapbox.io/tohoku-nlp/Miyawaki_AR

`bash train.sh -g [GPU_ID] -i [IN_DIR] -o [OUT_DIR]`
* `bash train.sh -g 1 -i ../PAS/NTC_Matsu_converted -o ./out` 


### 方針
* 入力ベクトルに次元４のラベルベクトルをconcat
* 得られたラベルベクトルに対して， 上位nまでを採用
<img src="https://latex.codecogs.com/gif.latex?n&space;=&space;N&space;\cdot&space;\frac{T-t}{T}" />
* 得られたラベルベクトルに対して， 入力ラベルベクトルに上書き
