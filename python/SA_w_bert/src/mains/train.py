import os
import sys
import json
import random
import logging
import warnings
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
        confusion_matrix, 
        accuracy_score, 
        precision_score,
        recall_score, 
        f1_score,
        classification_report
        )

from transformers import get_linear_schedule_with_warmup, AdamW

sys.path.append(os.getcwd())
from src.dataloaders.dataloader import SADataLoader, log_dataset
from src.models.sa_bert import SABert


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PAD = 0
Path = os.path.abspath

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if CUDA else 'cpu')
logger.info(f'cuda ... {CUDA}')
logger.info(f'device ... {DEVICE}')

sys.setrecursionlimit(3000)
warnings.simplefilter(action='ignore')

GREEN = '\033[32m'
YELLOW = '\033[33m'
END = '\033[0m'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_loop(args, model, dataloader, optimizer, criterion, scheduler):

    results = []
    best_f1, count_unimproved = 0, 0
    
    # logdir for tensorboard
    log_dir = os.path.join(args.dest, os.path.join(args.log_dir, model_id))
    os.makedirs(log_dir, exist_ok=True)
    
    # saved model dir
    dest_model = os.path.join(args.dest, 'models')
    os.makedirs(dest_model, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(args.epoch):
        logger.debug(f"EPOCH ... {epoch}/{args.epoch-1}")

        train_loss, train_scores = train_epoch(
                epoch,
                model,
                dataloader,
                optimizer,
                criterion,
                scheduler,
                )
        logger.debug(GREEN + '=== train ===' + END)
        logger.debug(GREEN + f"LOSS ... {train_loss}" + END)
        # logger.debug(f"Prec ... {round(train_scores['micro_prec'], 4)}")
        # logger.debug(f"Rec  ... {round(train_scores['micro_rec'], 4)}")
        logger.debug(f"F1   ... {round(train_scores['micro_f1'], 4)}")
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('Prec/train', train_scores['micro_prec'], epoch)
        writer.add_scalar('Rec/train', train_scores['micro_rec'], epoch)
        writer.add_scalar('F1/train', train_scores['micro_f1'], epoch)

        dev_loss, dev_scores = evaluate(
                epoch,
                model, 
                dataloader,
                criterion
                )
        logger.debug(YELLOW + "=== dev ===" + END)
        logger.debug(YELLOW + f"LOSS ... {dev_loss}" + END)
        # logger.debug(f"Prec ... {round(dev_scores['micro_prec'], 4)}")
        # logger.debug(f"Rec  ... {round(dev_scores['micro_rec'], 4)}")
        logger.debug(f"F1   ... {round(dev_scores['micro_f1'], 4)}")
        writer.add_scalar('loss/dev', dev_loss, epoch)
        writer.add_scalar('Prec/dev', dev_scores['micro_prec'], epoch)
        writer.add_scalar('Rec/dev', dev_scores['micro_rec'], epoch)
        writer.add_scalar('F1/dev', dev_scores['micro_f1'], epoch)

        if best_f1 < dev_scores['micro_f1']:
            count_unimproved = 0
            best_f1 = dev_scores['micro_f1']
            torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(), 
                        'scheduler': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'dev_loss': dev_loss,
                        'train_scores': train_scores,
                        'dev_scores': dev_scores
                       }, 
                       os.path.join(dest_model, 'best_model.pt'))
            logger.info(f'SAVE epoch ... {epoch}')
        else:
            # early stop
            if count_unimproved > args.early_stop:
                return results

            count_unimproved += 1
            checkpoint = torch.load(os.path.join(dest_model, 'best_model.pt'))
            model.load_state_dict(checkpoint['model'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            model = model.to(DEVICE)
        
        results.append(
                {   
                    'epoch': epoch,
                    'train_loss': round(train_loss, 6),
                    'train_prec': round(train_scores['micro_prec'], 6),
                    'train_rec': round(train_scores['micro_rec'], 6),
                    'train_f1': round(train_scores['micro_f1'], 6),
                    'dev_loss': round(dev_loss, 6),
                    'dev_prec': round(dev_scores['micro_prec'], 6),
                    'dev_rec': round(dev_scores['micro_rec'], 6),
                    'dev_f1': round(dev_scores['micro_f1'], 6),
                }
            )
    
    writer.close()
    logger.info("DONE train")
    return results


def train_epoch(epoch, model, dataloader, optimizer, criterion, scheduler):
    model.train()
    model.fix_parameters(requires_grad=True)
    storage = {'gold': [], 'pred': [], 'loss': []}
    loss = 0
    
    train_iterator = dataloader._create_iterator(dataloader._train) # shuffle
    pbar = tqdm(train_iterator)

    for idx, batch in enumerate(pbar):
        pbar.set_description(f'TRAIN ep.{epoch}')
        pbar.set_postfix(OrderedDict(loss=loss))

        input_ids, attention_mask, labels = list(map(lambda x: x.to(DEVICE), batch))
        texts = dataloader.decode(input_ids)

        if epoch == 0 and idx == 0:
            logger.info(f'SAMPLE input_ids[0] ... {input_ids[0]}')
            logger.info(f'SAMPLE attention_mask[0] ... {attention_mask[0]}')
            logger.info(f'SAMPLE labels[0] ... {labels[0]}')
            logger.info(f'SAMPLE texts[0] ... {texts[0]}')

        assert input_ids.size() == attention_mask.size()

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        import ipdb; ipdb.set_trace()

        #logits = model(input_ids, attention_mask=attention_mask)
        #loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        storage['loss'].append(loss.item())
        storage['gold'].extend(labels.tolist())
        storage['pred'].extend(torch.argmax(logits, -1).tolist())

    scores = calc_acc(storage['gold'], storage['pred'])
    return np.mean(storage['loss']), scores


def evaluate(epoch, model, dataloader, criterion):
    model.eval()
    model.fix_parameters(requires_grad=False)
    storage = {'gold': [], 'pred': [], 'loss': []}
    loss = 0

    dev_iterator = dataloader._create_iterator(dataloader._dev)
    
    pbar = tqdm(dev_iterator)
    for batch in pbar:
        pbar.set_description(f'EVAL ep.{epoch}')
        pbar.set_postfix(OrderedDict(loss=loss))

        input_ids, attention_mask, labels = list(map(lambda x: x.to(DEVICE), batch))
        texts = dataloader.decode(input_ids)

        with torch.no_grad():
            loss, logits = model(input_ids, labels=labels)
            # logits = model(input_ids, attention_mask)
            # loss = criterion(logits, labels)
        
        storage['loss'].append(loss.item())
        storage['gold'].extend(labels.tolist())
        storage['pred'].extend(torch.argmax(logits, -1).tolist())
    
    scores = calc_acc(storage['gold'], storage['pred'])
    return np.mean(storage['loss']), scores


def calc_acc(golds:list, preds:list) -> dict:
    # macro 平均: 各クラス毎に評価値を計算してから平均
    # micro-平均: 各クラス合計してから評価値を計算
    # macro << micro: １クラスの割当数が高く，正解率も高い，他は低い
    assert len(golds) == len(preds)
    #print(classification_report(golds, preds))  # 各クラスを陽性としたときの値と平均
    return {
        'confusion_matrix': confusion_matrix(golds, preds), # 縦:actual # 横:predicted
        'macro_prec': precision_score(golds, preds, average='macro'),
        'micro_prec': precision_score(golds, preds, average='micro'),
        'macro_rec': recall_score(golds, preds, average='macro'),
        'micro_rec': recall_score(golds, preds, average='micro'),
        'macro_f1': f1_score(golds, preds, average='macro'), 
        'micro_f1': f1_score(golds, preds, average='micro'),
        }


def create_model_id(args):
    global model_id
    model_id = f'{args.exp}_s{args.seed}_ep{args.epoch}_lr{args.lr}_mxl{args.max_len}_bs{args.bsize}_wup{args.warmup}'


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Train with BERT for Sentiment Analysis')
    parser.add_argument('-i', '--data_path', dest='fi', required=False, type=Path, default='result/data.src')
    parser.add_argument('-o', '--dest', dest='dest', required=False, type=str, default='work')
    parser.add_argument('-exp', type=str, required=True, help='exp name')
    parser.add_argument('--log_dir', type=str, default='runs', help='save at os.path.join(args.dest, args.log_dir)')
    parser.add_argument('--f_train', type=str, default=None)
    parser.add_argument('--f_dev', type=str, default=None)
    parser.add_argument('--f_test', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--max_token_len', dest='max_len', type=int, default=64)
    parser.add_argument('--data_size', dest='dsize', type=int, default=100)
    parser.add_argument('--batch_size', dest='bsize', type=int, default=4)
    parser.add_argument('--split_ratio', dest='ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='split ratio of dataset')
    parser.add_argument('--warmup', type=int, default=0, help='num of warmup steps for scheduler')
    parser.set_defaults(no_thres=False)
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    create_model_id(args)
    set_seed(args.seed)

    dataloader = SADataLoader(
            args.dest,
            f_merge = args.fi,
            f_train = args.f_train,
            f_dev = args.f_dev,
            f_test = args.f_test,
            data_size = args.dsize,
            batch_size = args.bsize,
            max_len = args.max_len,
            split_ratio = args.ratio,
            )

    model = SABert(dataloader, args.n_classes).to(DEVICE)
  
    optimizer = AdamW(
            model.parameters(), 
            lr=args.lr, 
            eps=1e-8,
            correct_bias=False)

    tmp_iterator = dataloader._create_iterator(dataloader._train)
    total_steps = len(tmp_iterator) * args.epoch
    
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup,
            num_training_steps=total_steps
            )

    criterion = nn.NLLLoss(
            ignore_index=PAD,
            ).to(DEVICE)
    
    results = train_loop(
                args,
                model,
                dataloader,
                optimizer,
                criterion,
                scheduler,
                )

    df = pd.DataFrame(results).set_index('epoch')
    df.to_csv(os.path.join(args.dest, 'results.csv'), 
            mode='w', sep='\t', float_format='%.3f')
    logger.info(f'SAVE results ... {args.dest}/results.csv')

    with open(os.path.join(args.dest, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    logger.info(f'SAVE configs ... {args.dest}/config.json')


if __name__ == '__main__':
    main()
    logger.info(f'DONE ... {__file__}')
