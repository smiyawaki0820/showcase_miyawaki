# Sentiment Analysis with BERT

## Create Data
```
$ python src/dataloaders/emoji_extract.py --input_file=絵文字を取り出したいデータファイル --output_dir=作成したデータを入れるディレクトリ --file_name=作成したデータのファイル名(扱うファイルが複数あるのでどのデータから取ってきたか識別できるような名前をつけてください)
```

## Data Path
```
/home/takumakato/Club-IMI-taiwa-2019/Japanese-sentiment-analysis/data/emoji_sentence.txt
```

## Data Analysis
```
/src/data_analysis/data_analysis.ipynb
```

## Training Code
``` running code
$ bash run.sh -g [GPU ID (opt)] -i [in_data (opt)] -o [dest_dir (opt)]
$ tensorboard --logdir [path of logs]

# dest_dir/ 下に以下が作成される
# - datasets/{train, dev, test}.csv
# - models/best_model.path
# - images/results.png
# - logs/[model_id]
# - config.json
# - results.csv

# python src/mains/train.py
# - input data として merged_data (`data_path`) もしくは splited_data (`f_train, f_dev, f_test`) を指定
# f_train, f_dev, f_test を指定した場合その使用が優先される
```

