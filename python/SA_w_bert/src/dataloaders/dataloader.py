import os
import sys
import logging
import argparse
import subprocess as subp
from pprint import pprint
from overrides import overrides
sys.path.append(os.getcwd())

import numpy as np
import torch
from torchtext import data
from transformers import BertJapaneseTokenizer
from sklearn.model_selection import train_test_split


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=int(os.environ['LOG_LEVEL']),
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

Path = os.path.abspath
dset = ('train', 'dev', 'test')
PRE_BERT = 'cl-tohoku/bert-base-japanese-char-whole-word-masking'


class MyDataset(data.Dataset):
    
    def __init__(self,
            dest: Path,             # destination of saved data
            f_merge: Path=None,     # merged csv data path
            f_train: Path=None,     # train csv data path
            f_dev: Path=None,       # dev csv data path
            f_test: Path=None,      # test csv data path
            data_size: int=100,
            split_ratio: list=[0.8, 0.1, 0.1],
            ):
        
        assert f_merge or all([x is not None for x in [f_train, f_dev, f_test]])

        self.ratio = split_ratio
        self.size = data_size
        self.dest_data = os.path.join(dest, 'datasets')
        os.makedirs(self.dest_data, exist_ok=True)
        
        # Tokenizer
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(PRE_BERT)
        # from janome.tokenizer import Tokenizer; Tokenize(wakati=True)

        # Field
        self._TEXT = data.Field(sequential=True, tokenize=self.tokenize)
        self._LABEL = data.LabelField(sequential=False)

        # dataset
        if all([x is not None for x in [f_train, f_dev, f_test]]):  # 優先
            self._train, self._dev, self._test = self.create(os.path.dirname(f_train), train=f_train, dev=f_dev, test=f_test)
        else:
            self._train, self._dev, self._test = self.create(os.path.dirname(f_merge), fi=f_merge)


    def tokenize(self, text:str) -> list:
        return self.tokenizer.tokenize(text)

    def create(self, ddir, **datasets):
        # 仕様: dataset には (train, dev, test) もしくは fi を指定
        def check_datasets(dsets):
            return all([dsets.get(x) for x in dset]) \
                    and tuple(dsets.keys()) == dset

        def split_data(dest, datasets):

            def write_lines(fo, lines: list):
                with open(fo, 'w') as f_csv:
                    for size, line in enumerate(lines):
                        print(line.rstrip(), file=f_csv)
                logger.debug(f'CREATE ... {fo} ({size} lines)')

            with open(datasets['fi'], 'r') as fi:
                lines = fi.readlines()
                lines = lines[:int(len(lines)*self.size/100)]
        
            n_lines = [int(len(lines)*r) for r in self.ratio]   # [80, 10, 10]

            # split
            train, test = train_test_split(
                    lines, test_size=n_lines[-1], shuffle=True, random_state=0)
            train, dev = train_test_split(
                    train, test_size=n_lines[1], shuffle=True, random_state=0)

            for fo, data in zip(dset, [train, dev, test]):  # save datasets
                write_lines(os.path.join(self.dest_data, datasets[fo]), data)

        ### main: create ###

        # need to split -> split_data
        if not check_datasets(datasets):
            assert datasets.get('fi'), 'DatasetError'
            for name in dset:
                datasets[name] = f'{name}.csv'
            split_data(self.dest_data, datasets)
        
        return data.TabularDataset.splits(
                path=self.dest_data, 
                train=os.path.basename(datasets['train']), 
                validation=os.path.basename(datasets['dev']), 
                test=os.path.basename(datasets['test']),
                format='tsv',
                fields=[('label',self._LABEL), ('text',self._TEXT)]
                )
    
 
class SADataLoader(MyDataset):

    def __init__(self,
            dest: Path,  # destination of saved data
            f_merge: Path=None,     # merged csv data path
            f_train: Path=None,     # train csv data path
            f_dev: Path=None,       # dev csv data path
            f_test: Path=None,      # test csv data path
            data_size: int=100,
            batch_size: int=1,
            max_len: int=32,
            split_ratio: list=[0.8, 0.1, 0.1],
            ):
        super().__init__(dest, f_merge, f_train, f_dev, f_test, data_size, split_ratio)
        self.bsize = batch_size
        self.mxlen = max_len

        # Vocabulary
        # self._TEXT.build_vocab(self._train)
        # self._LABEL.build_vocab(self._train)

    def _create_iterator(self, _dataset):

        def collate_fn(batch):
            try:
                texts = [data.text for data in batch if data]
                labels = torch.tensor([int(data.label) for data in batch if data]).squeeze()
            except:
                pprint([vars(data) for data in batch])
                exit
            texts = list(map(lambda x: self.encode(x), texts))

            # squeeze だと bsize=1 で 1D-Error
            input_ids = torch.stack([text['input_ids'] for text in texts]).squeeze()
            attention_mask = torch.stack([text['attention_mask'] for text in texts]).squeeze()
            return input_ids, attention_mask, labels    # tensor 以外 Error

        return torch.utils.data.DataLoader(
                _dataset,
                batch_size=self.bsize,
                shuffle=True,
                num_workers=2,  # num parallel
                collate_fn=collate_fn,
                )

    def encode(self, text:list):
        """ output ... input_ids + attention_mask """
        # https://huggingface.co/transformers/glossary.html#attention-mask
        
        return self.tokenizer.encode_plus(
                text,
                max_length=self.mxlen,
                add_special_tokens=True,        # [CLS] and [SEP]
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'             # return PyTroch tensor
            )

    def decode(self, ids: torch.tensor) -> list:
        ids = np.array(torch.squeeze(ids.cpu())).tolist()    # 2D
        return list(map(lambda x: self.tokenizer.convert_ids_to_tokens(x), ids))


def create_arg_parser():
    parser = argparse.ArgumentParser(description='create dataset')
    parser.add_argument('-i', '--data_path', dest='fi', required=False, type=Path,
            default='/work01/miyawaki/data/twitter_data_for_sentiment.data')
    parser.add_argument('-o', '--dest', dest='dest', required=False, type=str, default='work')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--data_size', dest='dsize', type=int, default=10)
    parser.add_argument('--batch_size', dest='bsize', type=int, default=1)
    parser.add_argument('--max_len', dest='max_len', type=int, default=32)
    parser.add_argument('--ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1], help="split ratio of datasets")
    parser.set_defaults(no_thres=False)
    return parser


def run():
    parser = create_arg_parser()
    args = parser.parse_args()

    dataloader = SADataLoader(
            args.dest,
            f_merge = args.fi,
            data_size = args.dsize,
            batch_size = args.bsize,
            max_len = args.max_len,
            split_ratio = args.ratio,
            )

    log_dataset(dataloader)

def log_dataset(dataloader):
    logger.debug(f"SIZE train ... {len(dataloader._train)}")
    logger.debug(f"SIZE dev ... {len(dataloader._dev)}")
    logger.debug(f"SIZE test ... {len(dataloader._test)}")
    logger.debug(f"SAMPLE train[0] ... {vars(dataloader._train[0])}")
    logger.info(f"PAD ... {dataloader.tokenizer.pad_token}")
    logger.info(f"UNK ... {dataloader.tokenizer.unk_token}")
    logger.info(f"CLS ... {dataloader.tokenizer.cls_token}")
    logger.info(f"SEP ... {dataloader.tokenizer.sep_token}")


if __name__ == '__main__':    
    run()
