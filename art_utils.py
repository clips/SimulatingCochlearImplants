import os
import csv
import numpy as np
from Models.utils import load_model
from sklearn.metrics import accuracy_score
from Featurize.utils import load_featurized
from ART.art import precision_recall_fscore_micro
from ART.art import labelingsignificance as art_test
from keras import backend as K
K.set_learning_phase(0)

file_dir = os.path.dirname(os.path.realpath(__file__))


def one_hot(x, nr_classes):
    x = x.astype(np.int32)
    out = np.zeros((x.shape[0], nr_classes))
    for idx, x in enumerate(x):
        out[idx, x] = 1
    return out


def get_pred(model, xs, nr_classes):
    pred = model.predict_classes(xs, batch_size=32, verbose=0)
    pred = one_hot(pred, nr_classes)
    return pred


def get_probas(model, xs):
    probas = model.predict(xs, batch_size=32, verbose=0)
    return probas


def get_medres_vs_lowres_outfile(filepath):
    if os.path.isfile(filepath):
        f = open(filepath, 'a')
        return f
    else:
        f = open(filepath, 'a')
        row_names = ['N',
                     'medres_acc',
                     'lowres_acc',
                     'diff',
                     'p',
                     'p_bonferroni']
        csv.writer(f).writerow((row_names))
    return f


def get_high_vs_lowres_outfile(filepath):
    if os.path.isfile(filepath):
        f = open(filepath, 'a')
        return f
    else:
        f = open(filepath, 'a')
        row_names = ['N',
                     'highres_acc',
                     'lowres_acc',
                     'diff',
                     'p',
                     'p_bonferroni']
        csv.writer(f).writerow((row_names))
    return f


def get_per_vs_cnn_outfile(filepath):
    if os.path.isfile(filepath):
        f = open(filepath, 'a')
        return f
    else:
        f = open(filepath, 'a')
        row_names = ['N',
                     'resolution',
                     'cnn_acc',
                     'per_acc',
                     'diff',
                     'p',
                     'p_bonferroni']
        csv.writer(f).writerow((row_names))
    return f


def med_vs_lowres(params, n_art):

    log = get_medres_vs_lowres_outfile(params['logpath'])

    low_xs, low_ys, y_to_idx, ids = load_featurized(params['lowres_path'])
    med_xs, med_ys, y_to_idx, ids = load_featurized(params['medres_path'])
    n_classes = len(y_to_idx)

    assert np.array_equal(low_ys, med_ys)
    ys = low_ys

    low_model = load_model(params['lowres_id'], params['model_dir'])
    med_model = load_model(params['medres_id'], params['model_dir'])

    low_preds = get_pred(low_model, low_xs, n_classes)
    med_preds = get_pred(med_model, med_xs, n_classes)

    low_acc = accuracy_score(ys, low_preds)
    med_acc = accuracy_score(ys, med_preds)

    p_diff = art_test(ys, low_preds, med_preds, absolute=True, n=n_art,
                      scoring=precision_recall_fscore_micro,
                      return_distribution=False)[2]

    r = [n_art, med_acc, low_acc, med_acc - low_acc,
         p_diff, p_diff * params['bonferroni']]

    csv.writer(log).writerow(r)
    log.close()


def high_vs_lowres(params, n_art):

    log = get_high_vs_lowres_outfile(params['logpath'])

    high_xs, high_ys, y_to_idx, ids = load_featurized(params['highres_path'])
    low_xs, low_ys, y_to_idx, ids = load_featurized(params['lowres_path'])
    n_classes = len(y_to_idx)

    assert np.array_equal(low_ys, high_ys)
    ys = low_ys

    high_model = load_model(params['highres_id'], params['highres_model_dir'])
    low_model = load_model(params['lowres_id'], params['lowres_model_dir'])

    high_preds = get_pred(high_model, high_xs, n_classes)
    low_preds = get_pred(low_model, low_xs, n_classes)

    low_acc = accuracy_score(ys, low_preds)
    high_acc = accuracy_score(ys, high_preds)

    p_diff = art_test(ys, low_preds, high_preds, absolute=True, n=n_art,
                      scoring=precision_recall_fscore_micro,
                      return_distribution=False)[2]

    r = [n_art, high_acc, low_acc, high_acc - low_acc,
         p_diff, p_diff * params['bonferroni']]

    csv.writer(log).writerow(r)
    log.close()


def per_vs_cnn(state, n_art):

    log = get_per_vs_cnn_outfile(state['logpath'])

    low_xs, low_ys, y_to_idx, ids = load_featurized(state['lowres_path'])
    med_xs, med_ys, y_to_idx, ids = load_featurized(state['medres_path'])
    n_classes = len(y_to_idx)

    assert np.array_equal(low_ys, med_ys)
    ys = med_ys

    per_medres = load_model(state['medres_id'], state['per_dir'])
    per_lowres = load_model(state['lowres_id'], state['per_dir'])
    cnn_medres = load_model(state['medres_id'], state['cnn_dir'])
    cnn_lowres = load_model(state['lowres_id'], state['cnn_dir'])

    preds_per_medres = get_pred(per_medres, med_xs, n_classes)
    preds_per_lowres = get_pred(per_lowres, low_xs, n_classes)
    preds_cnn_medres = get_pred(cnn_medres, med_xs, n_classes)
    preds_cnn_lowres = get_pred(cnn_lowres, low_xs, n_classes)

    acc_per_medres = accuracy_score(ys, preds_per_medres)
    acc_per_lowres = accuracy_score(ys, preds_per_lowres)
    acc_cnn_medres = accuracy_score(ys, preds_cnn_medres)
    acc_cnn_lowres = accuracy_score(ys, preds_cnn_lowres, n_classes)

    p_diff_ch = art_test(ys, preds_per_lowres, preds_cnn_lowres, absolute=True,
                         n=n_art, scoring=precision_recall_fscore_micro,
                         return_distribution=False)[2]

    p_diff_plain = art_test(ys, preds_per_medres, preds_cnn_medres,
                            absolute=True, n=n_art,
                            scoring=precision_recall_fscore_micro,
                            return_distribution=False)[2]

    r = [n_art, 'low_res', acc_cnn_lowres, acc_per_lowres,
         acc_cnn_lowres - acc_per_lowres,
         p_diff_ch, p_diff_ch * state['bonferroni']]
    csv.writer(log).writerow(r)

    r = [n_art, 'med_res', acc_cnn_medres, acc_per_medres,
         acc_cnn_medres - acc_per_medres,
         p_diff_ch, p_diff_plain * state['bonferroni']]

    csv.writer(log).writerow(r)
    log.close()
