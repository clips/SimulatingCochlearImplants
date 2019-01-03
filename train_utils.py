import os
from Featurize.utils import load_featurized
from Models.utils import train_with_repo, load_model


def get_train_valid(state):
    datapath = os.path.join(state['data_dir'], state['resolution'])
    train_path = datapath + '_train'
    valid_path = datapath + '_valid'
    x_train, y_train, label_to_idx, _ = load_featurized(train_path)
    x_valid, y_valid, label_to_idx, _ = load_featurized(valid_path)
    return x_train, y_train, x_valid, y_valid, label_to_idx


def train_model(get_model, params, params_dtypes, pre_trained_model=None):

    x_train, y_train, x_valid, y_valid, label_to_idx = get_train_valid(params)

    inputlength = x_train[0].shape[0]
    n_mels = x_train[0].shape[1]
    n_classes = len(label_to_idx)

    if not pre_trained_model:
        model = get_model(params, n_mels, inputlength, n_classes)
    else:
        model = pre_trained_model

    weightpath, n_epochs = train_with_repo(model, params, params_dtypes,
                                           x_train, y_train,
                                           x_valid, y_valid)
    return model, weightpath, n_epochs


def train_dummy_model(get_model, params, params_dtypes):
    params['resolution'] = 'hires'
    maxepochs = params['max_epochs']
    params['max_epochs'] = 1
    params['deafness_type'] = 'dummy_model'
    train_model(get_model, params, params_dtypes)
    params['max_epochs'] = maxepochs


def train_pd_models(get_model, params, params_dtypes):

    if not params['pretrained_dir']:
        params['deafness_type'] = 'normally_hearing'
        params['resolution'] = 'hires'
        train_model(get_model, params, params_dtypes)
        params['pretrained_dir'] = params['model_dir']
    else:
        params['deafness_type'] = 'dummy_model'
        train_dummy_model(get_model, params, params_dtypes)

    for r in ('medres', 'lores'):
        params['deafness_type'] = 'postlingually_deaf'
        params['resolution'] = r
        pretrained = load_model(0, params['pretrained_dir'])
        _, _, n_epochs = train_model(get_model, params, params_dtypes,
                                     pre_trained_model=pretrained)

def train_cd_models(get_model, params, params_dtypes):

    params['deafness_type'] = 'dummy_model'
    train_dummy_model(get_model, params, params_dtypes)

    for r in ('medres', 'lores'):
        params['deafness_type'] = 'congenitally_deaf'
        params['resolution'] = r
        _, _, n_epochs = train_model(get_model, params, params_dtypes)
