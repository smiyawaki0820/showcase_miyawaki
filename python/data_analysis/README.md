# 前処理

## カテゴリデータ処理
```pandas.py

# 要素の値を変換
df.sex = df.sex.replace(['male', 'female'], [0, 1])


```

## 欠損値処理
## 特徴量追加
## 次元数削減
* 主成分分析




# 手法選択
<img src="https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F329544%2F3959eff8-584f-9b3f-06e3-bc27a26cb8dc.png?ixlib=rb-1.2.2&auto=format&gif-q=60&q=75&w=1400&fit=max&s=eb1febce544af8fa5685df4c36291d0c">

<img src="https://datumstudio.jp/wp-content/uploads/2017/07/model_selection.png">

## 決定木モデル
* boosting
> 予測モデルの誤り予測に対して重みを加味しパラメータを更（前の学習結果を次の学習に反映）新．bias（未学習）を減らす．
* bagging (bootstrap aggregating)
> 学習データを復元抽出でランダム抽出し，学習を行う（それぞれのモデルを並列的に学習）．variance（過学習）を減らす

### XGBoost
* boosting．決定木の「階層」に着目（level-wise）
* 厳密な枝分かれポイントを探すため、全てのデータポイントを読み込む必要がある（`tree_method=hist` とすることで histgram-base のアルゴリズムを採用可能）

### LightGBM
[参考](https://www.codexa.net/lightgbm-beginner/)
* boosting．決定木の「葉」に着目（leaf-wise）
* 大規模なデータセットに対して計算コストを極力抑える
* 訓練データの特徴量を階級に分けてヒストグラム化することで、意図的に厳密な枝分かれを探さず大規模なデータセットに対しても計算コストを抑える


__特徴__
* モデル訓練にかかる時間が短い
* メモリ効率が高い
* 推測精度が高い（Leaf-Wiseの方がより複雑）
* 過学習しやすい
* 大規模データセットも訓練可能


<img src="https://www.codexa.net/wp-content/uploads/2019/02/DecisionTrees_2_thumb.png">


### Random Forest
* baging


# 参考
* [LightGBM](https://www.codexa.net/lightgbm-beginner/)
* [Benchmarking and Optimization of Gradient Boosting Decision Tree Algorithms](https://arxiv.org/abs/1809.04559)