import os
from art_utils import med_vs_lowres, high_vs_lowres, per_vs_cnn

file_dir = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":

    # number of data shuffles for approximate randomization testing
    N = 100000

    log_dir = os.path.join(file_dir, 'results', 'art')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    data_dir = os.path.join(file_dir, 'Featurize', 'featurized')
    models_dir = os.path.join(file_dir, 'results', 'models')

    highres_words = os.path.join(data_dir, 'words', 'hires_test')
    medres_words = os.path.join(data_dir, 'words', 'medres_test')
    lowres_words = os.path.join(data_dir, 'words', 'lores_test')

    highres_gender = os.path.join(data_dir, 'gender', 'hires_test')
    medres_gender = os.path.join(data_dir, 'gender', 'medres_test')
    lowres_gender = os.path.join(data_dir, 'gender', 'lores_test')

    params = dict()
    params['highres_id'] = 0
    params['medres_id'] = 1
    params['lowres_id'] = 2
    params['bonferroni'] = 18

    # run ART comparisons for word recognition
    ###########################################################################
    params['lowres_path'] = lowres_words
    params['medres_path'] = medres_words
    params['highres_path'] = highres_words

    params['logpath'] = os.path.join(log_dir, 'words_cd_cnn.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_cd_cnn')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_cd_per.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_cd_per')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_cd_cnn_0ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_cd_cnn_0ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_cd_cnn_1ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_cd_cnn_1ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_cd_per_0ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_cd_per_0ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_cd_per_1ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_cd_per_1ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_pd_cnn.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_pd_cnn')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_pd_per.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_pd_per')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_pd_cnn_0ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_pd_cnn_0ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_pd_cnn_1ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_pd_cnn_1ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_pd_per_0ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_pd_per_0ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'words_pd_per_1ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'words_pd_per_1ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'high_vs_lowres_words_per.csv')
    params['highres_model_dir'] = os.path.join(models_dir, 'words_pd_per')
    params['lowres_model_dir'] = os.path.join(models_dir, 'words_cd_per')
    high_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'high_vs_lowres_words_cnn.csv')
    params['highres_model_dir'] = os.path.join(models_dir, 'words_pd_cnn')
    params['lowres_model_dir'] = os.path.join(models_dir, 'words_cd_cnn')
    high_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'per_vs_cnn_words_pd.csv')
    params['per_dir'] = os.path.join(models_dir, 'words_pd_per')
    params['cnn_dir'] = os.path.join(models_dir, 'words_pd_cnn')
    per_vs_cnn(params, N)

    params['logpath'] = os.path.join(log_dir, 'per_vs_cnn_words_cd.csv')
    params['per_dir'] = os.path.join(models_dir, 'words_cd_per')
    params['cnn_dir'] = os.path.join(models_dir, 'words_cd_cnn')
    per_vs_cnn(params, N)

    # run ART comparisons for gender recognition
    ###########################################################################
    params['lowres_path'] = lowres_gender
    params['medres_path'] = medres_gender
    params['highres_path'] = highres_gender

    params['logpath'] = os.path.join(log_dir, 'gender_cd_cnn.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_cd_cnn')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_cd_per.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_cd_per')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_cd_cnn_0ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_cd_cnn_0ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_cd_cnn_1ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_cd_cnn_1ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_cd_per_0ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_cd_per_0ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_cd_per_1ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_cd_per_1ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_pd_cnn.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_pd_cnn')
    med_vs_lowres(params, N)
    
    params['logpath'] = os.path.join(log_dir, 'gender_pd_per.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_pd_per')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_pd_cnn_0ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_pd_cnn_0ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_pd_cnn_1ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_pd_cnn_1ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_pd_per_0ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_pd_per_0ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'gender_pd_per_1ep.csv')
    params['model_dir'] = os.path.join(models_dir, 'gender_pd_per_1ep')
    med_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'high_vs_lowres_gender_per.csv')
    params['highres_model_dir'] = os.path.join(models_dir, 'gender_pd_per')
    params['lowres_model_dir'] = os.path.join(models_dir, 'gender_cd_per')
    high_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'high_vs_lowres_gender_cnn.csv')
    params['highres_model_dir'] = os.path.join(models_dir, 'gender_pd_cnn')
    params['lowres_model_dir'] = os.path.join(models_dir, 'gender_cd_cnn')
    high_vs_lowres(params, N)

    params['logpath'] = os.path.join(log_dir, 'per_vs_cnn_gender_pd.csv')
    params['per_dir'] = os.path.join(models_dir, 'gender_pd_per')
    params['cnn_dir'] = os.path.join(models_dir, 'gender_pd_cnn')
    per_vs_cnn(params, N)

    params['logpath'] = os.path.join(log_dir, 'per_vs_cnn_gender_cd.csv')
    params['per_dir'] = os.path.join(models_dir, 'gender_cd_per')
    params['cnn_dir'] = os.path.join(models_dir, 'gender_cd_cnn')
    per_vs_cnn(params, N)
