# What is wandb ?
* 実験管理ツール

__pros__
* 機能の追加に用意に対応できる
  * UI はライブラリ管理していないため、常に最新状態のものが利用可能
* 複数台での管理が容易

__cons__
* ネットワーク環境がないと厳しい
* Standalone 版がない
  * クラウドやサービスを利用できない場面で問題になる

# Start
* [Weights & Biases](https://wandb.ai/site) にサインアップ

## Set up
```bash
! pip install wandb
```

# How to use
* [docs for Pytorch](https://docs.wandb.ai/integrations/pytorch)
* [full example (g-colab)](https://colab.research.google.com/drive/1QTIK23LBuAkdejbrvdP5hwBGyYlyEJpT?usp=sharing)
```python
import wandb

wandb.init(project="qasrl4ja")
config = {
  "optimizer": "Adam",
  "lr": 1e-3,
  "epochs": 20,
}
wandb.config.update(config)

# Logging gradients and model parameters
wandb.watch(model)
for bix, (data, gold) in enumerate(train_iter):
  ...
  if bix % args.log_interval == 0:
    wandb.log({"loss": loss})
```
