import os
import sys
import json

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris  

iris = load_iris()
df = pd.DataFrame(
    np.concatenate([iris.target.reshape(-1,1), iris.data], axis=-1),
    columns=(['label'] + iris.feature_names)
    )

df.label = df.label.astype(int)

fo = os.path.join(os.path.dirname(__file__), 'iris.csv')
df.to_csv(open(fo, 'w'), sep=',', index=False, header=True)
