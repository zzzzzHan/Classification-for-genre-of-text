#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:28:49 2021

"""

from tensorflow.keras.optimizers import Adam
import math 
from matplotlib import pyplot as plt
import data
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense,Dropout,GlobalAveragePooling1D,Embedding
import torch.nn as nn
import numpy as np
import keras
from keras.models import Model
#from keras.optimizers import SGD,Adam
from keras.layers import *
#from Attention_keras import Attention,Position_Embedding

class Self_Attention(keras.layers.Layer):

    def __init__(self, dimension_out, **kwargs):
        self.dimension_out = dimension_out
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, dimension_in):
        self.weight_matrix = self.add_weight(name='weight_matrix',
                                      shape=(3,dimension_in[2], self.dimension_out),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(dimension_in) 

    def call(self, x):
        Query = K.dot(x, self.weight_matrix[0])
        Key = K.dot(x, self.weight_matrix[1])
        Value = K.dot(x, self.weight_matrix[2])
        QK_T = K.batch_dot(Query,K.permute_dimensions(Key, [0, 2, 1]))
        normalized_QK_T = QK_T / (math.sqrt(length_sequense))
        transformed_QK_T = K.softmax(normalized_QK_T)
        final_value = K.batch_dot(transformed_QK_T,Value)
        return final_value


batch_size = 64
length_sequense = 300


def build_self_attention():
    '''Build the self attention model with pretrained embedding.
    '''
    sequences = Input(shape=(length_sequense,), dtype='int32')
    embeddings = Embedding(len(data.vocab_list), 300)(sequences)
    out = Self_Attention(300)(embeddings)
    out = GlobalAveragePooling1D()(out)
    out = Dropout(0.5)(out)
    outputs = Dense(4, activation='softmax')(out)
    model= Model(inputs=sequences, outputs=outputs)
    return model

model = build_self_attention()

print(model.summary())

#optimizer = Adam(lr=0.0002,decay=0.00001)
#optimizer = Adam(lr=0.001,decay=0.00001)
optimizer = 'adam'
#optimizer ='RMSprop'
#optimizer = 'sgd'
#optimizer = SGD(lr=0.01,momentum=0.9)
#loss='mse'
#loss='sparse_categorical_crossentropy'
loss = 'categorical_crossentropy'

# evaluation metrics 
def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = tp / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp / (positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision_1 = precision(y_true, y_pred)
    recall_1 = recall(y_true, y_pred)
    return 2*((precision_1*recall_1)/(precision_1+recall_1+K.epsilon()))

# compile the model
model.compile(loss=loss,optimizer=optimizer,metrics=['acc',f1,precision, recall])

print('Start to train')

model.fit(data.X_train,data.Y_train,
         batch_size=batch_size,
         epochs=4,
         validation_data=(data.validation_X,data.validation_Y))

predicted=model.predict(data.Xt_pad) 
result  =np.argmax(predicted,axis=1)
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(result): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()

