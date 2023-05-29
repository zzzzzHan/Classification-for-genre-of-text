#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
import nltk
from gensim.models import Word2Vec
import keras
import random 

# load the training data
preprocessing1 = False
preprocessing2 = True
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']

texts = {'X':X,
       'Y':Y}
data = pd.DataFrame(texts)
tests = {'Xt':Xt}
data_tests = pd.DataFrame(tests)

def split_sentence(sentence):
    '''split the sentence by punctuation
    '''
    stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sentence = re.sub(stop, '', sentence)
    return sentence.split()

sentences  = data.X.apply(split_sentence)
sentences_Xt  = data_tests.Xt.apply(split_sentence)

stem = PorterStemmer().stem
stopwords = set(nltk.corpus.stopwords.words("english"))
def process_document1(doc):
    '''Remove the stop words and apply the stemmer
    '''
    processed_doc = []
    for i in range(len(doc)):
        words_in_doc = doc[i]
        list_term = [t.lower() for t in words_in_doc]
        list_term = [stem(t)for t in list_term if t not in stopwords]
        processed_doc.append(list_term)
    return processed_doc


def process_document2(doc):
    '''Remove the stop words 
    '''
    processed_doc = []
    for i in range(len(doc)):
        words_in_doc = doc[i]
        list_term = [t.lower() for t in words_in_doc]
        list_term = [t for t in list_term if t not in stopwords]
        processed_doc.append(list_term)
    return processed_doc


if preprocessing1 ==True:
    sentences = process_document1(sentences)
    sentences_Xt = process_document1(sentences_Xt)
    
if preprocessing2 ==True:
    sentences = process_document2(sentences)
    sentences_Xt = process_document2(sentences_Xt)


embedding_vector_size = 300
# train the word 2 vector 
w2v_model = Word2Vec(
    sentences=sentences,vector_size=embedding_vector_size,
    min_count=1, window=3, workers=4)

# get the vocabulary list of the v2v 
vocab_list = list(w2v_model.wv.index_to_key)
# get the index that every word corresponding to 
word_index = {word: index for index, word in enumerate(vocab_list)}


def get_index(sentence):
    ''' Vectorize the sentence in the document by the index
    '''
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence
# get the vectorized document list 
X_data = list(map(get_index, sentences))
Xt_data = list(map(get_index, sentences_Xt))

# apply padding 
maxlen = 300
X_pad = pad_sequences(X_data, maxlen=maxlen)
Xt_pad = pad_sequences(Xt_data, maxlen=maxlen)

Y_value = data.Y.values
# get the one-hot lable of Y
one_hot_labels = keras.utils.np_utils.to_categorical(Y_value, num_classes=4)


def split_data():
    '''
    Split the original data set into trainnin and validation data set. The split is according to 
    the original proprotion of each class in the dataset. 
    '''
    group_obj = data.groupby('Y')
    group_0 = list(group_obj.groups[0]) 
    group_1 = list(group_obj.groups[1]) 
    group_2 = list(group_obj.groups[2])
    group_3 = list(group_obj.groups[3])
    num_0 = int(len(group_0)*0.1)
    num_1 = int(len(group_1)*0.1)
    num_2 = int(len(group_2)*0.1)
    num_3 = int(len(group_3)*0.1)
    validation_0 = random.sample(group_0, num_0) 
    validation_1 = random.sample(group_1, num_1) 
    validation_2 = random.sample(group_2, num_2)
    validation_3 = random.sample(group_3, num_3)    
    validation_index = validation_0+validation_1+validation_2+validation_3
    validation_X = [X_pad[i] for i in validation_index]
    validation_Y = [one_hot_labels[i] for i in validation_index]
    X_train = [X_pad[i] for i in range(len(Y))  if i not in validation_index]
    Y_train = [one_hot_labels[i] for i in range(len(one_hot_labels)) if i not in validation_index]
    return np.array(X_train),np.array(Y_train), np.array(validation_X),np.array(validation_Y)
 
 
#obtain the trainning and validation dataset  
X_train,Y_train, validation_X,validation_Y = split_data()






