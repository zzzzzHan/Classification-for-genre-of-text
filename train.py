#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:40:29 2021

@author: mac
"""
import json
import numpy as np

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import torch
from keras import utils as np_utils
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten, Conv1D,MaxPooling1D,Dropout,BatchNormalization,GlobalAveragePooling1D
from tensorflow.keras.optimizers import SGD
from keras.layers import LSTM
# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
num_term = tokenizer.word_index
vector = tokenizer.texts_to_sequences(X)
vector_padding = pad_sequences(vector,maxlen=620)

model = Sequential()
#model.add(Dense(32, activation='relu', input_dim=620))
#model.add(Dense(4, activation='softmax'))


#model.add(Embedding(41000, 300, input_length=620))       
model.add(Dense(64, activation='softmax', input_dim=620))
model.add(Dropout(0.5))
model.add(Dense(32, activation='softmax'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))


sgd = SGD(lr=0.05, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

#model.add(Dense(64, activation='relu', input_dim=620))
#model.add(Dropout(0.5))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(4, activation='softmax'))
#
#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.95, nesterov=True)
#model.compile(loss='categorical_crossentropy',
#              optimizer=sgd,
#              metrics=['accuracy'])




one_hot_labels = keras.utils.np_utils.to_categorical(Y, num_classes=4)
model.fit(vector_padding, one_hot_labels, epochs=20, batch_size=500)

tokenizer.fit_on_texts(Xt)
vector_test = tokenizer.texts_to_sequences(Xt)
vector_padding_test = pad_sequences(vector_test,maxlen=620)
Y_test = model.predict(np.array(vector_padding_test))
Y_test_pred = [np.argmax(i) for i in Y_test]

fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()
