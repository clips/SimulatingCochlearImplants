import os
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Models.ModelRepo import ModelRepo


def save_architecture(model, model_dir, model_id):
    # serialize model to JSON
    # note that weights need to be separately serialzed
    model_json = model.to_json()
    model_path = os.path.join(model_dir, '%s.json' % model_id)
    with open(model_path, 'w') as f:
        f.write(model_json)


def load_model(model_id, model_dir):
    # load json and create model
    model_path = os.path.join(model_dir, '%s.json' % model_id)
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # set trained weights
    weight_path = os.path.join(model_dir, '%s.h5' % model_id)
    model.load_weights(weight_path)
    return model


def train_with_repo(model, state, state_dtypes,
                    x_train, y_train, x_valid, y_valid):

    np.random.seed(state['random_seed'])
    repo = ModelRepo()
    repo.set_new_model(state, state_dtypes)
    weightpath = os.path.join(state['model_dir'], '%s.h5' % repo.model_id)

    checkpoint = ModelCheckpoint(weightpath, save_best_only=True,
                                 save_weights_only=True, monitor='val_loss',
                                 mode='auto', verbose=1)
    callbacks = [checkpoint]

    if state['early_stopping']:
        early = EarlyStopping(monitor='val_loss', patience=state['patience'],
                              verbose=1, mode='auto')
        callbacks.append(early)

    model.compile(loss=state['loss_function'],
                  optimizer=Adadelta(state['initial_learning_rate']),
                  metrics=['accuracy'])

    if state['max_epochs'] > 0:
        h = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                      batch_size=state['mini_batch_size'], verbose=1,
                      epochs=state['max_epochs'], callbacks=callbacks)

        n_epochs = len(h.history['loss']) - state['patience']
        repo.store_metrics_for_epochs(range(n_epochs), h.history)
        repo.conn.close()
    else:
        model.save_weights(weightpath, overwrite=True)
        n_epochs = 0
        repo.conn.close()

    save_architecture(model, state['model_dir'], repo.model_id)

    return weightpath, n_epochs
