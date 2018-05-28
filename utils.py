# from random import shuffle
import numpy as np
import pandas as pd
from numpy.random import shuffle


def get_data():
    # df = pd.read_csv('data/balance-scale.data')
    # df = df.loc[df['B'].isin(['L', 'R'])]
    #
    # df.B = pd.Categorical(df.B)
    # df['label'] = df.B.cat.codes
    # df = df.drop(['B'], axis=1)
    #
    # XY = df.as_matrix()
    # return XY[:,:-1], XY[:,-1:]


def batch(x, y, n):
    x = np.array(x)
    # y = np.array([y])
    y = np.array(y)
    # xy = np.concatenate((x, y.T), axis=1)
    xy = np.concatenate((x, y), axis=1)

    l = len(xy)
    shuffle(xy)
    for ndx in range(0, l, n):
        yield xy[ndx:min(ndx + n, l),:-1], xy[ndx:min(ndx + n, l),-1:]