`home/miyawaki_shumpei/YANS/showcase_miyawaki/yans2019/edit`

```実行
bash train_edit.sh -g [GPU_id] -i ~/PAS/NTC_Matsu_converted -o [output_dir]
```


# ./

## train_edit.sh / test.sh
- train/test 実行シェルスクリプト

## log_plot
- `./result/log/loss.txt`から loss をプロットするもの

## requirements.txt


# ./src

## train_edit.py
- main (モデル訓練)

## model_edit.py / bi_gru_for_srl.py
- モデル

## arg_edit.py
- 入力引数処理

## eval_edit.py
- test 

