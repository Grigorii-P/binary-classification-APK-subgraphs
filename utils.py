import numpy as np
import pickle
from numpy.random import shuffle

path_to_pca = '/home/pogorelov/pca_mat'
path_to_targets = '/home/pogorelov/targets'

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
    with open(path_to_pca, 'rb') as file:
        pca_mat = pickle.load(file)
    with open(path_to_targets, 'rb') as file:
        targets = pickle.load(file)

    X = np.array(pca_mat)
    Y = np.array(targets)
    return X, Y

def batch(x, y, n):
    x = np.array(x)
    y = np.reshape(np.array(y), newshape=(len(y),1))
    # xy = np.concatenate((x, y.T), axis=1)
    xy = np.concatenate((x, y), axis=1)

    l = len(xy)
    shuffle(xy)
    for ndx in range(0, l, n):
        yield xy[ndx:min(ndx + n, l),:-1], xy[ndx:min(ndx + n, l),-1:]