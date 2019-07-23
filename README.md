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


input data (train_edit.py)
- xss: 
    "https://gyazo.com/f6e8f380e3dce6e7c9fa1f95ff84f62b"
- yss: 
    "https://gyazo.com/9d422a75a420cd477d1e9dca64845e6c"
- score:
    "https://gyazo.com/7bbf1791e1d5cad0cb3003e73bd2b3d4"
    * ここの各列にラベルベクトルを
