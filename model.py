#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:38:31 2021

@author: mac
"""
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
import nltk
from gensim.models import Word2Vec
import keras
# load the training data

preprocessing = False

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
    stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sentence = re.sub(stop, '', sentence)
    return sentence.split()

sentences  = data.X.apply(split_sentence)
sentences_Xt  = data_tests.Xt.apply(split_sentence)

stem = nltk.PorterStemmer().stem
stopwords = set(nltk.corpus.stopwords.words("english"))
def process_document(doc):
    processed_doc = []
    for i in range(len(doc)):
        words_in_doc = doc[i]
        list_term = [t.lower() for t in words_in_doc]
        list_term = [stem(t)for t in list_term if t not in stopwords]
        processed_doc.append(list_term)
    return processed_doc

if preprocessing ==True:
    sentences = process_document(sentences)
    sentences_Xt = process_document(sentences_Xt)

"""
 训练Word2Vec
"""
# 嵌入的维度
embedding_vector_size = 300
w2v_model = Word2Vec(
    sentences=sentences,vector_size=embedding_vector_size,
    min_count=1, window=3, workers=4)

# 取得所有单词  
vocab_list = list(w2v_model.wv.index_to_key)
# 每个词语对应的索引
word_index = {word: index for index, word in enumerate(vocab_list)}


# 序列化
def get_index(sentence):
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence
X_data = list(map(get_index, sentences))
Xt_data = list(map(get_index, sentences_Xt))

# 截长补短
maxlen = 300
X_pad = pad_sequences(X_data, maxlen=maxlen)
Xt_pad = pad_sequences(Xt_data, maxlen=maxlen)
# 取得标签
Y_value = data.Y.values
one_hot_labels = keras.utils.np_utils.to_categorical(Y_value, num_classes=4)






class FastText(nn.Module):
    def __init__(self, vocab, w2v_dim, classes, hidden_size):
        super(FastText, self).__init__()
        #创建embedding
        self.embed = nn.Embedding(42000, w2v_dim)  #embedding初始化，需要两个参数，词典大小、词向量维度大小
        self.embed.weight.requires_grad = True #需要计算梯度，即embedding层需要被训练
        self.fc = nn.Sequential(              #序列函数
            nn.Linear(w2v_dim, hidden_size),  #这里的意思是先经过一个线性转换层
            nn.BatchNorm1d(hidden_size),      #再进入一个BatchNorm1d
            nn.ReLU(inplace=True),            #再经过Relu激活函数
            nn.Linear(hidden_size, classes)#最后再经过一个线性变换
        )

    def forward(self, x):                      
        x = self.embed(x)                     #先将词id转换为对应的词向量
        out = self.fc(torch.mean(x, dim=1))   #这使用torch.mean()将向量进行平均
        return out
    
def train_model(net, epoch, lr, data, label):      #训练模型
    print("begin training")
    net.train()  # 将模型设置为训练模式，很重要！
    optimizer = optim.Adam(net.parameters(), lr=lr) #设置优化函数
    Loss = nn.CrossEntropyLoss()  #设置损失函数
    for i in range(epoch):  # 循环
        optimizer.zero_grad()  # 清除所有优化的梯度
        output = net(data)  # 传入数据，前向传播，得到预测结果
        loss = Loss(output, label) #计算预测值和真实值之间的差异，得到loss
        loss.backward() #loss反向传播
        optimizer.step() #优化器优化参数

        # 打印状态信息
        print("train epoch=" + str(i) + ",loss=" + str(loss.item()))
    print('Finished Training')
    return net 
def model_test(net, test_data):
    #net.eval()  # 将模型设置为验证模式
    res = []
    with torch.no_grad():
        outputs = net(test_data)
        # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
        _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
        res.append(predicted)
        
if __name__ == "__main__":
    #这里没有写具体数据的处理方法，毕竟大家所做的任务不一样
    batch_size = 64
    epoch = 1  # 迭代次数
    w2v_dim = 300  # 词向量维度
    lr = 0.001
    hidden_size = 128
    classes = 4

    # 定义模型
    net = FastText(vocab=47200, w2v_dim=w2v_dim, classes=classes, hidden_size=hidden_size)

    # 训练
    print("开始训练模型")
    X_input = torch.tensor(np.array(vector_padding), dtype=torch.long)
    
    model = train_model(net, epoch, lr, X_input, Y_input)
    # 保存模型
    
    tokenizer.fit_on_texts(Xt)
    vector_test = tokenizer.texts_to_sequences(Xt)
    vector_padding_test = pad_sequences(vector_test,maxlen=620)
    testing_prediction = torch.tensor(np.array(vector_padding_test), dtype=torch.long)
    outputs = net(testing_prediction)
    pre =  torch.max(outputs.data, 1)


    fout = open("out.csv", "w")
    fout.write("Id,Y\n")
    for i, line in enumerate(pre): # Y_test_pred is in the same order as the test data
        fout.write("%d,%d\n" % (i, line))
    fout.close()
    
    