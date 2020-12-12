from preprocess import *
from evaluator import *
import numpy as np
#import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Model
from keras.layers import Input,Dense,LSTM,Dropout,Bidirectional,GRU,Permute,multiply
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
########读取图片、显示图片###########
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


seed = 2020
k=5
s=1
alphabet = 'ATCG'

nuc,link,len = fa2matrix(k,alphabet,path_1_H)
datasets,labels,input_length = datasets_labels(nuc,link)
print(np.sqrt(input_length))
datasets = datasets.reshape(-1,int(np.sqrt(input_length)),int(np.sqrt(input_length)))
TIME_STEPS = int(np.sqrt(input_length))
print(datasets.shape)
np.random.seed(seed=seed)

print(datasets)
print(labels)


mnist_input = Input(shape=(TIME_STEPS,TIME_STEPS),name='input')
lstm1 = Bidirectional(LSTM(32))(mnist_input)
hidden1 = Dense(64,activation='relu',name='hidden1')(lstm1)
drop = Dropout(0.5)(attention_mul)
output = Dense(1,activation='sigmoid',name='output')(drop)

kfold = StratifiedKFold(n_splits=20, random_state=seed, shuffle=True)
cvscores = []
cvSENS = []
cvSPEC = []
cvMCC = []
cvaucScore = []
i = 1
for train, validation in kfold.split(datasets, labels):
    print(i)
    i = i+1
    model = Model(inputs=mnist_input,outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.003), metrics=['accuracy',tf.keras.metrics.AUC()])
    model.fit(datasets[train], labels[train], epochs=32, batch_size=128, verbose=2,validation_data=(datasets[validation],labels[validation]),callbacks = [es])
    score = model.evaluate(datasets[validation], labels[validation], verbose=0)
    pred = model.predict(datasets[validation])
    pred = pred2label(pred)
    acc = accuracy(labels[validation], pred)
    cvscores.append(acc)
    recal = recall(labels[validation], pred)
    cvSENS.append(recal)
    Spec = spec(labels[validation], pred)
    cvSPEC.append(Spec)
    Mcc = mcc(labels[validation], pred)
    cvMCC.append(Mcc)
    cvaucScore.append(score[2])
    model = None

print('cvscores:%.3f (+/- %.2f)' % (np.mean(cvscores), np.std(cvscores)))
print('cvSENS:%.3f (+/- %.2f)' % (np.mean(cvSENS), np.std(cvSENS)))
print('cvSPEC:%.3f (+/- %.2f)' % (np.mean(cvSPEC), np.std(cvSPEC)))
print('cvMCC:%.3f (+/- %.2f)' % (np.mean(cvMCC), np.std(cvMCC)))
print('cvaucScore:%.3f (+/- %.2f)' % (np.mean(cvaucScore), np.std(cvaucScore)))


