# What is tensorboard ?
* 略


# How to Use
* [docs for Pytorch](https://pytorch.org/docs/stable/tensorboard.html)

```python
import torch
from torch.utils.tensorboard import SummaryWriter

exp_id = f'{model_id}_{dt.today().strftime("%Y%m%d-%H%M")}'
dir_tensorboard = os.path.join(dest, f'tensorboards/{exp_id}')
os.makedirs(dir_tensorboard, exist_ok=True)
logging.info(f'| CREATE ... {dir_tensorboard}')

with SummaryWriter(log_dir=dir_tensorboard, filename_suffix='retriever') as writer:
    min_loss, max_acc, best_epoch = np.inf, -np.inf, 0
    for epoch in epochs:
        scores: dict = train()
        writer.add_scalar(f'Loss/train', scores['train_loss'], epoch)
        writer.add_scalar(f'Loss/valid', scores['valid_loss'], epoch)
        writer.add_scalar(f'Acc/train',  scores['train_acc'],  epoch)
        writer.add_scalar(f'Acc/valid',  scores['valid_acc'],  epoch)
        writer.add_scalar(f'Av.Rank/valid', epoch_score['valid_ave_rank'], epoch)
        if scores['valid_loss'] < min_loss:
            min_loss = scores['valid_loss']
            max_acc  = scores['valid_acc']
            best_epoch = 0
            
    writer.add_hparams(
            {
                'seed':         args.seed, 
                'batch':        args.batch_size, 
                'lr':           args.learning_rate, 
                'warmup_steps': args.warmup_steps, 
                'dropout':      args.dropout
            },
            {
                'valid_loss': min_loss, 
                'valid_acc':  max_acc,
                'best_epoch': best_epoch
            }
        )

logging.info(f'| tensorboard --logdir {dir_tensorboard} --bind_all')
```
