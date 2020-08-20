import os
import sys

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


fi_data = os.path.abspath('../datasets/iris.csv')


def principal_component_analysis(features, targets, n_components=2, fo_img='pca.png'):
    
    # 主成分分析
    pca = PCA(n_components=n_components)
    pca.fit(features)

    # 分析結果を元にデータセットを主成分に変換
    transformed = pca.fit_transform(features)
    
    for label in np.unique(targets):
        plt.scatter(transformed[targets == label, 0],
                    transformed[targets == label, 1])
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')

    # 主成分の寄与率を出力する
    print('各次元の寄与率: {}'.format([round(e, 3) for e in pca.explained_variance_ratio_]))
    print('累積寄与率: {:.3f}'.format(sum(pca.explained_variance_ratio_)))

    plt.savefig(fo_img, bbox_inches='tight')


def main():
    df = pd.read_csv(open(fi_data))
    cols = df.columns.to_list()
    
    principal_component_analysis(df[cols[1:]].values, df.label.values)

if __name__ == '__main__':
    main()