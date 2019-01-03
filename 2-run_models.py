import os
import shutil
from Models.models import cnn, perceptron
from train_utils import train_pd_models, train_cd_models

file_dir = os.path.abspath(os.path.dirname(__file__))

# set parameters (initialize with None if they are variable)
# as well as data types for SQL data base with model results

params = dict()         # map parameter names to parameter values
params_dtypes = dict()  # map parameter names to parameters types
# (for storage in SQLITE data base)

params['max_epochs'] = None
params['mini_batch_size'] = None
params['n_hidden_dense'] = None
params['dropout'] = None
params['loss_function'] = 'categorical_crossentropy'
params['random_seed'] = 100
params['initial_learning_rate'] = None
params['early_stopping'] = True
params['patience'] = 10
params['resolution'] = None
params['deafness_type'] = None

# data types of parameters for storage in SQLITE data base
params_dtypes['max_epochs'] = 'INTEGER'
params_dtypes['mini_batch_size'] = 'INTEGER'
params_dtypes['n_hidden_dense'] = 'INTEGER'
params_dtypes['dropout'] = 'REAL'
params_dtypes['loss_function'] = 'TEXT'
params_dtypes['random_seed'] = 'INTEGER'
params_dtypes['initial_learning_rate'] = 'REAL'
params_dtypes['early_stopping'] = 'INTEGER'
params_dtypes['patience'] = 'INTEGER'
params_dtypes['resolution'] = 'TEXT'
params_dtypes['deafness_type'] = 'TEXT'


# helper functions

def remove_file_if_exists(filepath):
    if os.path.isfile(filepath):
        print('File already exists: %s' % filepath)
        print('Will overwrite it.')
        os.remove(filepath)


def remove_dir_if_exists(dir):
    if os.path.isdir(dir):
        print('Directory already exists: %s' % dir)
        print('Will delete and re-create it.')
        shutil.rmtree(dir)
        os.makedirs(dir)


def train_per(pretrain, data_name):
    params['data_dir'] = os.path.join(file_dir, 'Featurize', 'featurized',
                                      data_name)
    db_dir = os.path.join(file_dir, 'results', 'data_bases')
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    params['data_base_path'] = os.path.join(db_dir, '%s.sqlite' % params['id'])
    remove_file_if_exists(params['data_base_path'])
    params['model_dir'] = os.path.join(file_dir, 'results',
                                       'models', params['id'])
    remove_dir_if_exists(params['model_dir'])

    if not os.path.exists(params['model_dir']):
        os.makedirs(params['model_dir'])

    params['n_hidden_dense'] = None
    params['dense_dropout'] = None
    params['kernel_size1'] = None
    params['kernel_size2'] = None
    params['number_filters'] = None
    params['stride1'] = None
    params['stride2'] = None
    params['conv_dropout'] = None
    params['mini_batch_size'] = 32
    params['initial_learning_rate'] = 0.01

    if pretrain:
        train_pd_models(perceptron, params, params_dtypes)
    else:
        train_cd_models(perceptron, params, params_dtypes)


def train_cnn(pretrain, data_name):
    params['data_dir'] = os.path.join(file_dir, 'Featurize', 'featurized',
                                      data_name)
    db_dir = os.path.join(file_dir, 'results', 'data_bases')
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    params['data_base_path'] = os.path.join(db_dir, '%s.sqlite' % params['id'])
    remove_file_if_exists(params['data_base_path'])
    params['model_dir'] = os.path.join(file_dir, 'results',
                                       'models', params['id'])
    remove_dir_if_exists(params['model_dir'])

    if not os.path.exists(params['model_dir']):
        os.makedirs(params['model_dir'])

    params['n_hidden_dense'] = 100
    params['dense_dropout'] = 0.5
    params['number_filters'] = 5
    params['kernel_size1'] = 5
    params['kernel_size2'] = 5
    params['stride1'] = 2
    params['stride2'] = 2
    params['conv_dropout'] = 0.1
    params['mini_batch_size'] = 100
    params['initial_learning_rate'] = 0.1

    if pretrain:
        train_pd_models(cnn, params, params_dtypes)
    else:
        train_cd_models(cnn, params, params_dtypes)


if __name__ == '__main__':

    # 1. CNN
    ##########################################################################

    # gender recognition, congenitally deaf (CD) models
    params['id'] = 'gender_cd_cnn'
    params['max_epochs'] = 9999999
    train_cnn(pretrain=False, data_name='gender')

    params['id'] = 'gender_cd_cnn_1ep'
    params['max_epochs'] = 1
    train_cnn(pretrain=False, data_name='gender')

    params['id'] = 'gender_cd_cnn_0ep'
    params['max_epochs'] = 0
    train_cnn(pretrain=False, data_name='gender')

    # gender recognition, postlingually deaf (PD) models
    params['id'] = 'gender_pd_cnn'
    params['max_epochs'] = 9999999
    params['pretrained_dir'] = None
    train_cnn(pretrain=True, data_name='gender')

    params['id'] = 'gender_pd_cnn_1ep'
    params['max_epochs'] = 1
    params['pretrained_dir'] = os.path.join(file_dir, 'results', 'models',
                                            'gender_pd_cnn')
    train_cnn(pretrain=True, data_name='gender')

    params['id'] = 'gender_pd_cnn_0ep'
    params['max_epochs'] = 0
    params['pretrained_dir'] = os.path.join(file_dir, 'results', 'models',
                                            'gender_pd_cnn')
    train_cnn(pretrain=True, data_name='gender')

    # word recognition, congenitally deaf (CD) models
    params['id'] = 'words_cd_cnn'
    params['max_epochs'] = 9999999
    train_cnn(pretrain=False, data_name='words')

    params['id'] = 'words_cd_cnn_1ep'
    params['max_epochs'] = 1
    train_cnn(pretrain=False, data_name='words')

    params['id'] = 'words_cd_cnn_0ep'
    params['max_epochs'] = 0
    train_cnn(pretrain=False, data_name='words')

    # word recognition, postlingually deaf (PD) models
    params['id'] = 'words_pd_cnn'
    params['max_epochs'] = 9999999
    params['pretrained_dir'] = None
    train_cnn(pretrain=True, data_name='words')

    params['id'] = 'words_pd_cnn_1ep'
    params['max_epochs'] = 1
    params['pretrained_dir'] = os.path.join(file_dir, 'results', 'models',
                                            'words_pd_cnn')
    train_cnn(pretrain=True, data_name='words')

    params['id'] = 'words_pd_cnn_0ep'
    params['max_epochs'] = 0
    params['pretrained_dir'] = os.path.join(file_dir, 'results', 'models',
                                            'words_pd_cnn')
    train_cnn(pretrain=True, data_name='words')

    # 1. perceptron
    ###########################################################################

    # gender recognition, congenitally deaf (CD) models
    params['id'] = 'gender_cd_per'
    params['max_epochs'] = 9999999
    train_per(pretrain=False, data_name='gender')

    params['id'] = 'gender_cd_per_1ep'
    params['max_epochs'] = 1
    train_per(pretrain=False, data_name='gender')

    params['id'] = 'gender_cd_per_0ep'
    params['max_epochs'] = 0
    train_per(pretrain=False, data_name='gender')

    # gender recognition, postlingually deaf (PD) models
    params['id'] = 'gender_pd_per'
    params['max_epochs'] = 9999999
    params['pretrained_dir'] = None
    train_per(pretrain=True, data_name='gender')

    params['id'] = 'gender_pd_per_1ep'
    params['max_epochs'] = 1
    params['pretrained_dir'] = os.path.join(file_dir, 'results', 'models',
                                            'gender_pd_per')
    train_per(pretrain=True, data_name='gender')

    params['id'] = 'gender_pd_per_0ep'
    params['max_epochs'] = 0
    params['pretrained_dir'] = os.path.join(file_dir, 'results', 'models',
                                            'gender_pd_per')
    train_per(pretrain=True, data_name='gender')

    # word recognition, congenitally deaf (CD) models
    params['id'] = 'words_cd_per'
    params['max_epochs'] = 9999999
    train_per(pretrain=False, data_name='words')

    params['id'] = 'words_cd_per_1ep'
    params['max_epochs'] = 1
    train_per(pretrain=False, data_name='words')

    params['id'] = 'words_cd_per_0ep'
    params['max_epochs'] = 0
    train_per(pretrain=False, data_name='words')

    # word recognition, postlingually deaf (PD) models
    params['id'] = 'words_pd_per'
    params['max_epochs'] = 9999999
    params['pretrained_dir'] = None
    train_per(pretrain=True, data_name='words')

    params['id'] = 'words_pd_per_1ep'
    params['max_epochs'] = 1
    params['pretrained_dir'] = os.path.join(file_dir, 'results', 'models',
                                            'words_pd_per')
    train_per(pretrain=True, data_name='words')

    params['id'] = 'words_pd_per_0ep'
    params['max_epochs'] = 0
    params['pretrained_dir'] = os.path.join(file_dir, 'results', 'models',
                                            'words_pd_per')
    train_per(pretrain=True, data_name='words')
