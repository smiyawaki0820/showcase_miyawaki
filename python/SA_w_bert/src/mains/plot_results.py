import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set()

logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='plot from results.csv')
    parser.add_argument('--fi', type=str, help='path of results.csv')
    parser.set_defaults(no_thres=False)
    return parser


def plot_results(df, dest):
    fig = plt.figure(figsize=(18, 12))
    for i, tgt in enumerate(['loss', 'prec', 'rec', 'f1'], start=1):
        df_tmp = df[[f'train_{tgt}', f'dev_{tgt}']].rename(columns={f'train_{tgt}':'train', f'dev_{tgt}':'dev'})
        ax = fig.add_subplot(2, 2, i)
        df_tmp.plot(ax=ax, title=tgt)
        ax.set_xlabel('epoch')
        ax.set_ylabel(tgt)

    fo = os.path.join(dest, 'results.png')
    plt.savefig(fo, bbox_inches='tight')
    logger.info(f'CREATE ... {fo}')


def run():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    dest = os.path.dirname(os.path.abspath(args.fi))
    df = pd.read_csv(args.fi, delim_whitespace=True).set_index('epoch')
    plot_results(df, dest)


if __name__ == '__main__':
    run()
    logger.info(f'DONE ... {__file__}')
