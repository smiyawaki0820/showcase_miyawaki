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
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

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


class Models(object):
    def __init__(self, fi_csv):
        self.data = pd.read_csv(fi_csv)
        self.cols = self.data.columns.to_list()
        rate = [.8, .1, .1]

        train_X, self.test_X, train_y, self.test_y = \
            train_test_split(self.data[self.cols[1:]], self.data.label, shuffle=True, test_size=rate[2],)
        self.train_X, self.dev_X, self.train_y, self.dev_y = \
            train_test_split(train_X, train_y, shuffle=True, test_size=(rate[1]/(rate[0]+rate[1])),)

        assert all(map(lambda x,y: x.shape[0] == y.shape[0], [self.train_X, self.dev_X, self.test_X], [self.train_y, self.dev_y, self.test_y]))
    

    def my_xgboost(self):
        import xgboost as xgb
        
        train_data = xgb.DMatrix(self.train_X, label=self.train_y)
        dev_data = xgb.DMatrix(self.dev_X, label=self.dev_y)
        test_data = xgb.DMatrix(self.test_X)

        n_round = 1000
        early_stop = 5
        params = {
            'max_depth': 4,                 # max_depth of tree
            'eta': 0.3,                     # lr
            'objective': 'multi:softmax',   # objective
            'num_class': 3,                 # n_classes
            'eval_metric': 'mlogloss'       # dev metric
        }

        # train
        evals = [(dev_data, 'eval'), (train_data, 'train')]
        bst = xgb.train(params, train_data, n_round, evals, early_stopping_rounds=early_stop)
        logging.info('Best Score:{0:.4f}, Iteratin:{1:d}, Ntree_Limit:{2:d}'.format(bst.best_score, bst.best_iteration, bst.best_ntree_limit)) 

        # test
        pred = bst.predict(test_data, ntree_limit=bst.best_ntree_limit)

        # evaluate
        score = metrics.accuracy_score(self.test_y, pred)
        logging.info('TestAcc:{0:.4f}'.format(score)) 

        # ax = xgb.plot_importance(bst)
        # ax.figure.savefig('importance.png', bbox_inches='tight')
        # logger.info(f'plot | importance.png')
        # ax = xgb.plot_tree(bst)
        # ax.figure.savefig('tree.png', bbox_inches='tight')
        # logger.info(f'plot | tree.png')

    def my_linear_regression(self):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(self.train_X, self.train_y)

        logging.info('決定係数(train): {:.3f}'.format(model.score(self.train_X,self.train_y)))
        logging.info('決定係数(test): {:.3f}'.format(model.score(self.test_X, self.test_y)))
        logging.info('切片: {:.3f}'.format(model.intercept_))

        logging.info('正解率(train): {:.3f}'.format(model.score(self.train_X, self.train_y)))
        logging.info('正解率(dev): {:.3f}'.format(model.score(self.dev_X, self.dev_y)))
        logging.info('正解率(dev): {:.3f}'.format(model.score(self.test_X, self.test_y)))
        logging.info(f'オッズ比: {np.exp(model.coef_)}')

    def my_svm(self):
        from sklearn.svm import LinearSVC
        model = LinearSVC()
        model.fit(self.train_X, self.train_y)

        logging.info('正解率(train):{:.3f}'.format(model.score(self.train_X, self.train_y)))
        logging.info('正解率(dev):{:.3f}'.format(model.score(self.dev_X, self.dev_y)))
        logging.info('正解率(test):{:.3f}'.format(model.score(self.test_X, self.test_y)))

    def my_knn(self):
        from sklearn.neighbors import  KNeighborsClassifier
        train_acc, dev_acc, test_acc = [], [], []
        N = 100
        for n_neighbors in range(1,N):
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(self.train_X, self.train_y)
            train_acc.append(model.score(self.train_X, self.train_y))
            dev_acc.append(model.score(self.dev_X, self.dev_y))
            test_acc.append(model.score(self.test_X, self.test_y))

        plt.plot(range(1,N), train_acc, label='train')
        plt.plot(range(1,N), dev_acc, label='dev')
        plt.plot(range(1,N), test_acc, label='test')
        plt.ylabel('Accuracy')
        plt.xlabel('n_neighbors')
        plt.legend()
        plt.savefig('knn_scores.png', bbox_inches='tight')


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    models = Models(args.fi_iris)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
