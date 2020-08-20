import os
import sys
import glob
import pickle
import logging
import argparse
import tempfile
from pprint import pprint
from typing import List, Generator
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import optuna
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split

logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=int(os.environ.get('LOG_LEVEL', 30)),
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Q-CtrPrediction-Entry No.3')
    parser.add_argument('-fo', required=True, type=str, help='submission file consisted of [id, click]')
    parser.add_argument('--ddir', default='../data', type=str, help='data dir')
    parser.add_argument('--fs_train', default='train/*', type=str, help='train files')
    parser.add_argument('--fs_test', default='test/*', type=str, help='test files')
    parser.add_argument('--f_category', default='category.csv', type=str, help='category file name')
    parser.add_argument('--verbose', default=10, type=int, help='verbose_eval for LGB')
    parser.add_argument('--n_boost_round', default=50, type=int, help='num of grad_boost iterations for LGB')
    parser.add_argument('--early_stop', default=5, type=int, help='early stop for LGB')
    parser.add_argument('--n_trials', default=1, type=int, help='num of trial for optuna')
    # parser.add_argument('--n_kf', default=5, type=int, help='num of k-fold')
    parser.set_defaults(no_thres=False)
    return parser


class MyDataset(object):
    def __init__(self,
                ddir: str,
                f_category: str,
                fs_train: str,
                fs_test: str,
                ):

        self.data_files = {
            'train': glob.glob(os.path.join(ddir, fs_train)),
            'test': glob.glob(os.path.join(ddir, fs_test))
        }

        self.f_category = os.path.join(ddir, f_category)
        self.category_dict = {k:v.split() for k,v in list(map(lambda x: x.strip().split(','), open(self.f_category).readlines()[1:]))}
        # self.df_category = pd.read_csv(self.f_category).set_index('article_id')

        self.header = {
            'train': self.get_header(self.data_files['train'][0]).split(','),
            'test': self.get_header(self.data_files['test'][0]).split(',')
        }

        self.features = ['id', 'user_id', 'campaign_id', 'device', 'os', 'browser', 'main_category', 'sub_category', 'month', 'hour']

    def read_files(self, paths: List[str]) -> Generator[str, None, None]:
        for file_path in paths:
            for i, line in enumerate(open(file_path)):
                if i:
                    yield line.rstrip('\n')

    def get_header(self, file_path: str) -> str:
        return next(open(file_path)).rstrip('\n')

    def convert_article_into_category(self, article_id, main=False, sub=False) -> str:
        """ article_id を指定すると main_category, sub_category を返す """
        # return self.df_category.at[int(article_id), 'categories']
        return self.category_dict[article_id][0] if main else self.category_dict[article_id][0] if sub else None

    def create_df_fm_readfiles(self, data_type='train') -> pd.DataFrame:
        """ read_files より読み込んだ一行をヘッダーと対応付けた dictl を作成し，pd.DataFrame に変換 """
        li = [{h:l for h, l in zip(self.header[data_type], line.split(','))} for line in self.read_files(self.data_files[data_type])]
        return pd.DataFrame(li)

    def create_dataset(self, df, set_type='train') -> tuple:
        """ 訓練データの特徴量エンジニアリング結果を X として返却 """
        df['device'] = df.device.replace(['mobile', 'pc'], [0, 1])
        df['os'] = df.os.replace(['ios', 'mac', 'android', 'win'], [0, 1, 2, 3])
        df['browser'] = df.browser.replace(['firefox', 'ie', 'default', 'edge', 'chrome', 'safari'], [0, 1, 2, 3, 4, 5])

        # split category
        ds_main_category = df.article_id.map(lambda x: self.convert_article_into_category(x, main=True))
        ds_sub_category = df.article_id.map(lambda x: self.convert_article_into_category(x, sub=True))
        li_main_categories = list(set(ds_main_category))
        li_sub_categories = list(set(ds_sub_category))
        df['main_category'] = ds_main_category.replace(li_main_categories, range(len(li_main_categories)))
        df['sub_category'] = ds_sub_category.replace(li_sub_categories, range(len(li_sub_categories)))

        # split datetime
        # df['datetime'] = df.datetime.map(lambda x: pd.to_datetime(x))     # 遅い
        # df['day'] = df.datetime.map(lambda x: x.day)
        df['month'] = df.datetime.map(lambda x: int(x[5:7]))                # 仮定
        df['hour'] = df.datetime.map(lambda x: int(x[11:13]))
    
        X = df[self.features].astype(int)
        y = df['click'].astype(int) if set_type == 'train' else \
            pd.Series(-np.ones(df.shape[0]), name='click', dtype=int) if set_type == 'test' else None

        return X, y


class MyModel():
    def __init__(self):
            
        # 予め ハイパラ探索 を行った 
        self.params = {
                'objective': 'binary',
                'metric': 'auc',
                'lambda_l1': 0.03254155475596874, 
                'lambda_l2': 0.017113850407315497, 
                'feature_fraction': 0.4794925808547898, 
                'bagging_fraction': 0.4320563090720952, 
                'max_bin': 274, 
                'learning_rate': 0.034994754239373274, 
                'num_leaves': 20255, 
                'max_depth': 11, 
                'min_child_samples': 22
                }
        self.categorical_features = ['user_id', 'device', 'os', 'browser', 'main_category', 'sub_category']
        

    def lgbm(self, args, X_train, y_train, X_eval, y_eval, X_test=None, y_test=None, isSearch=True):
        
        global best_auc
        best_auc = 0.0
        tmp_dir = tempfile.TemporaryDirectory()

        def objective(trial):
            if isSearch:
                self.params['lambda_l1'] = trial.suggest_loguniform('lambda_l1', 8e-3, 1.0)
                # self.params['lambda_l2'] = trial.suggest_loguniform('lambda_l2', 5e-3, 1e-1)
                # self.params['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.4, 0.6)
                # self.params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.4, 0.6)
                # self.params['max_bin'] = trial.suggest_int('max_bin', 200, 300)
                self.params['learning_rate'] = trial.suggest_loguniform('learning_rate', 5e-3, 1e-1)
                self.params['num_leaves'] = trial.suggest_int('num_leaves', int(.7*2**10), int(.7*2**16))
                self.params['max_depth'] = trial.suggest_int('max_depth', 10, 16)
                self.params['min_child_samples'] = trial.suggest_int('min_child_samples', 20, 80)

            lgb_train = lgb.Dataset(X_train, y_train,
                                    categorical_feature=self.categorical_features)
            lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train,
                                    categorical_feature=self.categorical_features)
                                    
            logger.info(f'TRAIN label contrib ... 1: {(y_train==1).sum()}, 0: {(y_train==0).sum()}')
            logger.info(f'EVAL label contrib ... 1: {(y_eval==1).sum()}, 0: {(y_eval==0).sum()}')

            model = lgb.train(self.params, lgb_train,
                            valid_sets=lgb_eval,
                            verbose_eval=int(args.verbose),
                            num_boost_round=int(args.n_boost_round),
                            early_stopping_rounds=int(args.early_stop), 
                            )

            y_pred = model.predict(X_eval, num_iteration=model.best_iteration)

            fpr, tpr, _ = metrics.roc_curve(y_eval, y_pred)
            auc = metrics.auc(fpr, tpr)
            logger.info(f'auc ... {auc}')

            if best_auc < auc:
                with open(os.path.join(tmp_dir.name, 'best_model.pkl'), 'wb') as fp:
                    pickle.dump(model, fp)

            return auc

        logger.info(f'### Train ###')
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0), direction='maximize')
        study.optimize(objective, n_trials=args.n_trials)
        logger.debug(f'BEST params ... {study.best_params}')
        self.params.update(study.best_params)

        logger.info(f'### Predict ###')
        if (X_test is not None) and (y_test is not None):
            with open(os.path.join(tmp_dir.name, 'best_model.pkl'), 'rb') as f:
                model = pickle.load(f)
            
            y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            ds_test_pred = pd.Series(y_test_pred, name='click')
            pd.concat([X_test.id, ds_test_pred], axis=1).to_csv(args.fo, header=True, index=False)
            logger.info(f'CREATE ... {args.fo}')

        tmp_dir.cleanup()


def k_fold(X_train, y_train, k=5, shuffle=True, seed=0) -> Generator:
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    for train_index, eval_index in kf.split(X_train):
        yield (
            X_train.loc[train_index, :],     # cv_X_train
            X_train.loc[eval_index, :],      # cv_X_eval
            y_train.loc[train_index, :],     # cv_y_train
            y_train.loc[eval_index, :],      # cv_y_eval
        )

# 使わない
def pca_transform(X):
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(X)
    transformed = pca.fit_transform(X)
    return pd.DataFrame(transformed, columns=X.columns)


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    dataloader = MyDataset(
            args.ddir,
            args.f_category, 
            args.fs_train,
            args.fs_test,
    )

    df_train = dataloader.create_df_fm_readfiles('train')
    df_test = dataloader.create_df_fm_readfiles('test')

    l0_df_train = (df_train.click=='0').sum()
    l1_df_train = (df_train.click=='1').sum()
    logger.debug(f'df_train click==1 rate ... {l1_df_train/df_train.shape[0]}')

    per = 1.00  # click==1 を全体の 10% にするよう，click==0 をサンプリング（軽くしたい）
    frac = ((1-per)/per) * (l1_df_train/l0_df_train)

    logger.debug(f'drop click==0 rate ... {frac}')

    # 重いので under sampling 的なことをする
    df_train = pd.concat([df_train[df_train['click'] == '0'].sample(frac=frac, random_state=0), df_train[df_train['click'] == '1']])      

    logger.info(f'### Create Dataset ###')
    X_train, y_train = dataloader.create_dataset(df_train, 'train')
    X_test, y_test = dataloader.create_dataset(df_test, 'test')         # y_test: initialized -1

    logger.info(f'### Loading Model ###')
    model = MyModel()

    sp_X_train, sp_X_valid, sp_y_train, sp_y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

    """ k-fold
    for kf_idx, (cv_X_train, cv_X_eval, cv_y_train, cv_y_eval) in enumerate(k_fold(X_train, y_train, k=args.n_kf)):
        logger.info(f'KF ... {kf_idx}/{args.n_kf}')
        model.train(args, cv_X_train, cv_y_train, cv_X_eval, cv_y_eval)
    """

    model.lgbm(args, 
               sp_X_train, sp_y_train,
               sp_X_valid, sp_y_valid,
               X_test=X_test, y_test=y_test,
               isSearch=True
               )


if __name__ == '__main__':
    """
    trainに含まれる閲覧ログと、category.csvを学習用データとして用い、
    あるユーザーがある記事内のあるキャンペーンをクリックする確率を予測するモデルを作成してください。 
    これを用いて、testに含まれる閲覧ログに対するクリック確率を予測してください。
    """
    main()
