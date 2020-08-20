import os
import sys
import json
import logging
import argparse
from pprint import pprint
from typing import List, Generator
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from tqdm import tqdm

import optuna

import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split

logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='XGBoost against iris')
    parser.add_argument('--fi_iris', type=str, default='../../datasets/iris.csv')
    parser.set_defaults(no_thres=False)
    return parser


class XGBoost(object):
    def __init__(self, fi_csv):
        self.data = pd.read_csv(fi_csv)
        self.cols = self.data.columns.to_list()
        self.rate = [.8, .1, .1]

        self.train_data, self.dev_data, self.test_data, self.test_y = \
            self.data_reader(data=self.data, cols=self.cols, rate=self.rate)

        self.n_round = 1000
        self.early_stop = 5
        self.params = {
            'max_depth': 4,                 # max_depth of tree
            'eta': 0.3,                     # lr
            'objective': 'multi:softmax',   # objective
            'num_class': 3,                 # n_classes
            'eval_metric': 'mlogloss'       # dev metric
        }

    def set_params():
        booster_params = {
            'eta': 0.3,             # lr 
            'gamma': 0,             # [0,∞): 決定木の葉の追加による損失減少の下限（大きいほど保守的）
            'max_depth': 6,         # [0,∞): max depth
            'min_child_weight': 1,  # [0,∞): leaf weight の下限
            'max_delta_step': 0,    # [0,∞): restriction
            'lambda': 1,            # L2
            'alpha': 0,             # L1
        }
        learning_params = {
            'objective': 'reg:linear',   # minimizing loss_func
            # reg:linear   ... 線形回帰
            # reg:logistic ... ロジスティック回帰
            # binary:logistic ... 二値分類で確率を返す
            # multi:softmax   ... 多値分類でクラスの値を返す（num_classesを指定）
        }


    def data_reader(self, data=None, cols=None, rate=[0.8, 0.1, 0.1]):
        """ data format
        * data: pd.DataFrame
        * columns: list = ['label'] + [features]
        """
        data = data if data is not None else self.data
        cols = cols if cols is not None else self.cols

        train_X, test_X, train_y, test_y = \
            train_test_split(data[cols[1:]], data.label, shuffle=True, test_size=rate[2],)
        train_X, dev_X, train_y, dev_y = \
            train_test_split(train_X, train_y, shuffle=True, test_size=(rate[1]/(rate[0]+rate[1])),)

        assert train_X.shape[0] == train_y.shape[0]
        assert dev_X.shape[0] == dev_y.shape[0]
        assert test_X.shape[0] == test_y.shape[0]
        
        train_data = xgb.DMatrix(train_X, label=train_y)
        dev_data = xgb.DMatrix(dev_X, label=dev_y)
        test_data = xgb.DMatrix(test_X)
        return train_data, dev_data, test_data, test_y

    def run_xgboost(self):
        # train
        bst = self.train(
            params = self.params, 
            train_data = self.train_data, 
            dev_data = self.dev_data,
            n_round = self.n_round,
            )

        logging.info('Best Score:{0:.4f}, Iteratin:{1:d}, Ntree_Limit:{2:d}'.format(bst.best_score, bst.best_iteration, bst.best_ntree_limit)) 

        # test
        pred = self.predict(bst, test_data=self.test_data)

        # evaluate
        score = self.evaluate(pred, test_y=self.test_y)
        logging.info('TestAcc:{0:.4f}'.format(score)) 

        # plot
        self.plot_importance(bst, 'importance.png')

    def train(self, params=None, train_data=None, dev_data=None, n_round=None):
        params = params if params is not None else self.params
        train_data = train_data if train_data is not None else self.train_data
        dev_data = dev_data if dev_data is not None else self.dev_data
        n_round = n_round if n_round is not None else self.n_round

        evals = [(self.dev_data, 'eval'), (self.train_data, 'train')]
        return xgb.train(params, train_data, n_round, evals, early_stopping_rounds=self.early_stop)

    def predict(self, bst, test_data=None):
        test_data = test_data if test_data is not None else self.test_data
        return bst.predict(test_data, ntree_limit=bst.best_ntree_limit)

    def evaluate(self, pred, test_y=None):
        test_y = test_y if test_y is not None else self.test_y
        return metrics.accuracy_score(test_y, pred)

    def plot_importance(self, bst, fo_img):
        """ どの特徴量が予測結果に影響したか可視化 """
        if fo_img is None: return
        ax = xgb.plot_importance(bst)
        ax.figure.savefig(fo_img, bbox_inches='tight')
        logger.info(f'plot | {fo_img}')

    def plot_tree(self, bst, fo_img):
        """ どの特徴量が予測結果に影響したか可視化 """
        if fo_img is None: return
        ax = xgb.plot_tree(bst)
        ax.figure.savefig(fo_img, bbox_inches='tight')
        logger.info(f'plot | {fo_img}')


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    model = XGBoost(args.fi_iris)
    
    bst = model.train()
    logging.info('Best Score:{0:.4f}, Iteratin:{1:d}, Ntree_Limit:{2:d}'.format(bst.best_score, bst.best_iteration, bst.best_ntree_limit)) 

    pred = model.predict(bst)

    score = model.evaluate(pred)
    logging.info('TestAcc:{0:.4f}'.format(score)) 

    model.plot_importance(bst, 'importance.png')
    model.plot_tree(bst, 'tree.png')


if __name__ == '__main__':
    main()
