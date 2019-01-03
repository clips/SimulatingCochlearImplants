import os
import json
import numpy as np
from CorpusReaders.TIMITReader import TIMITReader
from CorpusReaders.GSCReader import GSCReader
from Featurize.featurize import featurize

file_dir = os.path.abspath(os.path.dirname(__file__))


def to_file(xs, ys, y_to_idx, ids, data_dir, resolution, partition):
    fn = '%s_%s' % (resolution, partition)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    path = os.path.join(data_dir, fn+'.json')
    json.dump([ys, y_to_idx, ids], open(path, 'w'))
    np.save(os.path.join(data_dir, fn+'.npy'), xs)


if __name__ == '__main__':

    # featurize TIMIT data
    timit_dir = os.path.join(file_dir, 'TIMIT', 'timit')
    reader = TIMITReader(timit_dir)
    ds_name = 'gender'
    save_to = os.path.join(file_dir, 'Featurize', 'featurized', ds_name)

    n_hires = 32
    n_medres = 16

    utts = reader.get_gender_utterances(0.2, 0.2)
    feats, y_to_idx = featurize(utts, n_hires, n_medres, sr=16000,
                                cut_to_mean_len=True)

    for resolution, data in feats.items():

        to_file(data['x_train'], data['y_train'], y_to_idx,
                data['train_ids'], save_to, resolution, 'train')
        to_file(data['x_valid'], data['y_valid'], y_to_idx,
                data['valid_ids'], save_to, resolution, 'valid')
        to_file(data['x_test'], data['y_test'], y_to_idx,
                data['test_ids'], save_to, resolution, 'test')

    # featurize GoogleSpeechCommands data
    corpus_dir = os.path.join(file_dir, 'GSC')
    reader = GSCReader(corpus_dir)
    ds_name = 'words'
    save_to = os.path.join(file_dir, 'Featurize', 'featurized', ds_name)

    n_hires = 32
    n_medres = 16

    words = reader.get_train_valid_test(20, 20, aux_words=True)
    feats, y_to_idx = featurize(words, n_hires, n_medres, sr=16000,
                                cut_to_mean_len=False)

    for resolution, data in feats.items():
        to_file(data['x_train'], data['y_train'], y_to_idx,
                data['train_ids'], save_to, resolution, 'train')
        to_file(data['x_valid'], data['y_valid'], y_to_idx,
                data['valid_ids'], save_to, resolution, 'valid')
        to_file(data['x_test'], data['y_test'], y_to_idx,
                data['test_ids'], save_to, resolution, 'test')