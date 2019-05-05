import sys
import numpy as np

import sys

from keras.layers.convolutional import *
from keras import initializers
import random
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pickle
from keras import backend
from keras.models import Model
from keras import optimizers
from keras.layers import *
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation
from keras_contrib.layers import CRF
from keras_contrib.metrics.crf_accuracies import *
from keras_contrib.losses.crf_losses import *
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import KFold
from utils import *
import keras
K = keras.backend

def load_sepsis_model():
    K.clear_session()
    num_features = 40
    input_x = Input(shape=(None, num_features))
    # x = input_x
    x = GaussianNoise(0.01)(input_x)

    n_feature_maps = num_features
    # print('build conv_x')
    # conv_x = keras.layers.normalization.BatchNormalization()(x)
    # conv_x = Activation('relu')(conv_x)
    # conv_x = Dropout(0.2)(conv_x)
    # conv_x = Conv1D(n_feature_maps, 25, strides=1, padding="same")(conv_x)
    # # conv_x = MaxPooling1D(padding='same')(conv_x)
    # conv_x = keras.layers.normalization.BatchNormalization()(conv_x)

    # # shortcut_y = MaxPooling1D(padding='same')(x)
    # shortcut_y = x
    # print('Merging skip connection')
    # y = add([shortcut_y, conv_x])

    # print("Y output shaspe ", y.get_shape())
    y = x
    for k in range(6):
        # print("k = ", k)
        # print('build conv_x')
        x1 = y
        conv_x = keras.layers.normalization.BatchNormalization()(x1)
        conv_x = Activation('relu')(conv_x)
        conv_x = Dropout(0.5)(conv_x)
        conv_x = Conv1D(n_feature_maps, 12, strides=1, padding="same",kernel_regularizer=regularizers.l2(0.05))(conv_x)
        # if k == 0:
        #     conv_x = MaxPooling1D(padding='same')(conv_x)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)
        # conv_x = Conv1D(n_feature_maps, 6, strides=1, padding="same")(conv_x)

        # conv_x = Conv1D(n_feature_maps, 6, strides=1, padding="same")(conv_x)
        # # if k == 0:
        # #     conv_x = MaxPooling1D(padding='same')(conv_x)
        # conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        # conv_x = Activation('relu')(conv_x)
        # conv_x = Dense(n_feature_maps)(conv_x)
        # # if k == 0:
        # #     conv_x = MaxPooling1D(padding='same')(conv_x)
        # conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        # conv_x = Activation('relu')(conv_x)

        # conv_x = LSTM(units=n_feature_maps, return_sequences=True, recurrent_dropout=0.15)(conv_x)
        conv_x = Bidirectional(LSTM(
            units=n_feature_maps/2, return_sequences=True, recurrent_dropout=0.5, kernel_regularizer=regularizers.l2(0.05)))(conv_x)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        # conv_x = Bidirectional(LSTM(units=n_feature_maps/2, return_sequences=True, recurrent_dropout=0.5) )(conv_x)
        # conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        # conv_x = Activation('relu')(conv_x)
        # shortcut_y = MaxPooling1D(padding='same')(x1)
        shortcut_y = x1
        # print('Merging skip connection')
        y = add([shortcut_y, conv_x])

    # x = Bidirectional(LSTM(units=8, return_sequences=True, activation='tanh',
    #                                 recurrent_dropout=0.5) )(y)  # variational biLSTM
    # x = Bidirectional(LSTM(units=32, return_sequences=True, activation='tanh',
    #                                 recurrent_dropout=0.5) )(x)
    # y = x
    # for i in range()
    # x = BatchNormalization()(x)

    # for i in range(5):
    #     x1 = x
    #     # x1 = BatchNormalization()(x1)
    #     x1 = Bidirectional(LSTM(units=20, return_sequences=True,
    #                                 recurrent_dropout=0.5) )(x1)
    #     x = add([x1, x])
        # x = BatchNormalization()(x)

    # x =
    # # x = TimeDistributed(Dense(128,activation='softmax'))(x)
    # x = TimeDistributed(Dense(20,activation='relu',kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = TimeDistributed(Dense(20,activation='relu',kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = TimeDistributed(Dense(100,kernel_initializer=keras.initializers.he_uniform(seed=None),
    #     activation='relu',kernel_regularizer=regularizers.l2(0.001))) (x)
    # x = Dropout(0.1)(x)
    # x = Dense(40,kernel_initializer='random_uniform',
    #     activation='relu', kernel_regularizer=regularizers.l2(0.001))(y)

    # x = Dense(20,kernel_initializer='random_uniform',
    #     activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = y

    # x = TimeDistributed(Dense(100,kernel_initializer='random_uniform',
    #     activation='relu', kernel_regularizer=regularizers.l2(0.001)))(x)

    # x = TimeDistributed(Dense(80,kernel_initializer='random_uniform',
    #     activation='relu'
    #     # , kernel_regularizer=regularizers.l1(0.001)

    #     ))(x)

    # # x = TimeDistributed(Dense(40,kernel_initializer='random_uniform',
    # #     activation='relu'
    # #     , kernel_regularizer=regularizers.l2(0.01)

    # #     ))(x)

    # x = Dropout(0.8)(x)
    # # x = TimeDistributed(Dense(40,kernel_initializer=keras.initializers.he_uniform(seed=None),
    # #     activation='relu', kernel_regularizer=regularizers.l2(0.001)))(x)
    # # x = Dropout(0.1)(x)
    # x = TimeDistributed(Dense(20,kernel_initializer='random_uniform',
    #     activation='relu'
    #     # , kernel_regularizer=regularizers.l1(0.001)
    #     ))(x)
    crf = CRF(2, sparse_target=False, learn_mode='marginal')  # CRF layer
    out = crf(x)  # output
    # out = Dense(1, activation='sigmoid')(x)
    # model = Model(input_x, out)

    # x = y

    # x = Dropout(0.2)(x)

    # out = TimeDistributed(Dense(2, kernel_initializer='random_uniform',
    #                             activation='softmax'
    #                             ))(x)

    model_list = []
    for k_fold in range(5):
        model = Model(input_x, out)

        adam = optimizers.Adam(lr=0.001, epsilon=1e-5,
                            clipvalue=0.8, amsgrad=True)
        model.compile(optimizer=adam, loss=crf_loss, metrics=[], sample_weight_mode = 'temporal')
        model.load_weights(filepath='best_model_fold_'+str(k_fold)+'.ckpt')
        model_list.append(model)


    


    # #read challenge data
    # data = get_data_from_file(input_file)
    # X_test, _ = prepare_input_for_lstm_crf([data], is_training=False)

    # #normalize test data
    # min_data = np.load('min_data.txt.npy')
    # max_data = np.load('max_data.txt.npy')

    # for idx, t_sequence in enumerate(X_test):
    #     X_test[idx] = (t_sequence - min_data) / \
    #         (max_data - min_data + 1e-8)


    # for k_fold in range(5):

    #     model.load_weights(filepath='best_model_fold_'+str(k_fold)+'.ckpt')
    #     if k_fold == 0:
    #         scores = model.predict(np.array(X_test[0]).reshape((1, len(X_test[0]), 40)))[0]
    #     else:
    #         scores += model.predict(np.array(X_test[0]).reshape((1, len(X_test[0]), 40)))[0]
    # return scores/5.0    
    return model_list




def get_sepsis_score(data, model_list):
    #read challenge data
    #data = get_data_from_file(input_file)
    # X_test, _ = prepare_input_for_lstm_crf([data], is_training=False)

    #impute missing data

    data = impute_missing_data(data)
    X_test = [data]
    # print(X_test)
    #normalize test data
    min_data = np.load('min_data.txt.npy')
    max_data = np.load('max_data.txt.npy')

    # print(min_data)
    # print(max_data)

    for idx, t_sequence in enumerate(X_test):
        X_test[idx] = (t_sequence - min_data) / \
            (max_data - min_data + 1e-8)
    
    for k_fold in range(5):

        # model_list[k_fold].load_weights(filepath='best_model_fold_'+str(k_fold)+'.ckpt')
        if k_fold == 0:
            scores = model_list[k_fold].predict(np.array(X_test[0]).reshape((1, len(X_test[0]), 40)))[0]
        else:
            scores += model_list[k_fold].predict(np.array(X_test[0]).reshape((1, len(X_test[0]), 40)))[0]
        # print(scores)
    scores = scores/5.0
    current_score = scores[-1][1]
    if current_score >=0.14:
        label = 1
    else:
        label = 0
    return current_score, label
