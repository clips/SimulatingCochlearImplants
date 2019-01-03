import json
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.utils import np_utils


def load_featurized(datapath):
    ys, y_to_idx, ids = json.load(open(datapath + '.json', 'r'))
    xs = np.load(datapath + '.npy')
    ys = np_utils.to_categorical(np.asarray(ys), len(y_to_idx))
    return xs, ys, y_to_idx, ids
