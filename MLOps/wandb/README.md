# What is wandb ?
* 実験管理ツール


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
