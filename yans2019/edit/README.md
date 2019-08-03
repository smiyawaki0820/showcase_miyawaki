`home/miyawaki_shumpei/YANS/showcase_miyawaki/yans2019/edit`

- 実行 (train)
```
bash train_edit.sh -g [GPU_id] -i ~/PAS/NTC_Matsu_converted -o [output_dir]
```


# 変更点

- labelベクトル として score を 次epoch で使用する
    - 高い値の score を json ファイルとして出力

```code:./work/score.json
  {
     "train/950101-0001-950101004.ntc":[
       {"6" (sent_id): [
         {"batch_id": "0", "word_idx": "", "label": [ga, wo, ni, null]},
         {"batch_id": "1", "word_idx": "", "label": [ga, wo, ni, null]}
         ]
       },
       {"23": [
         {"batch_id": "0", "word_idx": [ga, wo, ni, null]}
         ]
       }
     ],
     "train/950101-0002-950101008.ntc":[
       {"4": [
         {"batch_id": "0", "word_idx": [ga, wo, ni, null]}
         ]
       }
     ]
   }
```

- これを load することで， 入力ラベルベクトルとして定義
```
import ijson

if os.path.exists('./work/score.json'):
 
    with open('./work/score.json', 'rb') as jsf:
        objects = ijson.items(jsf, str(file_name[0])+'.item')
        ob = (o[str(sent_id[0])] for o in objects if o.get(str(sent_id[0])))
        for lis in ob:
            for dic in lis:
                temp_label = [float(i) for i in dic["label"]]
                temp[int(dic["batch_id"]),int(dic["word_idx"]),:] = torch.from_numpy(np.array(temp_label))
                
### IN:{word, pred, score(t-1)}, OUT:{score(t)} ###
scores = model(xss, temp)
```






# ./

### train_edit.sh / test.sh
- train/test 実行シェルスクリプト

### log_plot
- `./result/log/loss.txt`から loss をプロットするもの

### requirements.txt


# ./src

### train_edit.py
- main (モデル訓練)

### model_edit.py / bi_gru_for_srl.py
- モデル

### arg_edit.py
- 入力引数処理

### eval_edit.py
- test 

