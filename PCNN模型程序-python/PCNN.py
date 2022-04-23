#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:39:11 2017

@author: mark
"""

import numpy as np
import keras
import time
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from keras.layers import Conv1D
from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input

from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

import random

#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold

#from keras.utils import plot_model

Cro_v = 10     #交叉验证的折数
nb_ep = 120     #训练的轮数
nb_bsize = 4    #batch size 的大小
drop_percent = 0.32
cross_time = 1
print_command = 1    #1：打印细节；0不打印
nb_sample = 360


sub = 9  #被试的编号
#numpara = [1,6,11,16,21,26]  #对应的参数对
numpara = [1]  #对应的参数对

Results_Table = PrettyTable(["Subject Number","Parameter Number", "Test Loss", "Test Accuracy"])
#Results_Table.align["Parameter Number"] = "l"# Left align city names
Results_Table.padding_width = 1 # One space between column edges and contents (default)

start_time = time.time()
print time.strftime("%m-%d (%H+8):%M", time.localtime())

for k1 in numpara:
    
    sum_loss = 0
    sum_acc = 0
    for t in xrange(cross_time):
        #print t+1, "cross val --------------------------"
        print "Subject:",sub,";----", "Parameter:",k1,";----", "Cross_Time:",t+1,"."
        
        cross_loss = 0
        cross_acc = 0

        '''
        read sample data from file
        '''
        #file_path = "/notebooks/data/Train_p30/s" + str(k1) + "/p30/"
        #file_path = "/notebooks/eegdata/energy_0416/s" + str(k1) + "/Train/p3/"
        file_path = "/notebooks/data_new/CSPSTFT2/Train/s" + str(sub) + "/"
        X = []
        for i in xrange(1, nb_sample+1):      
            f = open(file_path + str(i) + ".csv")
            temp = []
            for line in f.readlines():
                temp.append(np.array([float(a[::]) for a in line.split(",")]))
                #temp.append(np.array([float(a[:6]) for a in line.split(",")]))
            X.append(np.array(temp))
        X = np.array(X)
        #print X.shape
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

        #X0 = np.zeros((((360,93,32,2)))
        #a0 = np.append(X1,X2,axis=0)

        X = np.array(X)
        X = X.reshape(nb_sample, 32, 32)

        '''
        read label from file
        '''

        #file_path = "/notebooks/data/Label_p30/s" + str(k1) + "/"
        #file_path = "/notebooks/eegdata/energy_0416/s" + str(k1) + "/Label/"
        file_path = "/notebooks/data_new/CSPSTFT2/Label/s" + str(sub) + "/"
        f = open(file_path + "label.csv")

        #Y = [int(a) for a in f.readline().spilt(",")]
        #Y = [f.readlines()]
        #Y_=f.readline().split(',')
        Y_ = f.readline().split(',')
        Y_[-1] = Y_[-1].strip("\n")
        Y = [int(a) for a in Y_]
        Y = [a-1 for a in Y]  #put label value from [1,2] to [0,1]
        Y = np.array(Y)
        #print Y.shape

        '''
        将0~359随机分成K组，每组 360/K 个数
        '''
        #a = random.randint(0, 100000)
        #a = 13812
        a = 98078
        #print '--------------------------- seed:',a
        np.random.seed(a)

        A = range(nb_sample)
        np.random.shuffle(A)
        A = np.array(A)
        B = np.split(A, Cro_v, axis = 0)

        '''
          for j=1,2,...,k
              将除第j份的所有数据作为训练集用于训练，得到训练参数。
              将训练参数在第j份数据上进行测试，得到测试错误E（j）
        '''


        for v in xrange(0, Cro_v):
            #将对应索引下的训练数据取出来一份作为测试集，其余作为训练集；标签数据集同
            traindata = []
            trainlabel =[]
            testdata = []
            testlabel =[]

            for j in xrange(0, Cro_v):

                C = B[j]

                if j == v:
                    for k2 in xrange(0, (nb_sample/Cro_v)):
                        d = C[k2]
                        testdata.append(X[d])
                        testlabel.append(Y[d])

                else:
                    for k2 in xrange(0, (nb_sample/Cro_v)):
                        d = C[k2]
                        traindata.append(X[d])
                        trainlabel.append(Y[d])

            #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

            testlabel = np.array(testlabel)
            testdata = np.array(testdata)
            trainlabel = np.array(trainlabel)
            traindata = np.array(traindata)        

            X_train = traindata.reshape(-1, 1, 32, 32)
            X_test = testdata.reshape(-1, 1, 32, 32)
            y_train = np_utils.to_categorical(trainlabel, num_classes= 2)
            y_test = np_utils.to_categorical(testlabel, num_classes= 2)

            #def baseline_model():

            inp = Input(shape=(1,32,32))
            
            ###   convlution in time   ###
            out_time = Conv2D(16,(1,3), 
                         kernel_initializer='TruncatedNormal',
                         bias_initializer='zeros',
                         strides=1, padding='same', activation='selu')(inp)
            out_time = BatchNormalization()(out_time)
            out_time = Dropout(drop_percent)(out_time)
            
            out_time = Conv2D(32,(1,3), strides=2, padding='same', activation='selu')(out_time)
            out_time = BatchNormalization()(out_time)
            out_time = Dropout(drop_percent)(out_time)
            
            out_time = Conv2D(64,(1,3), strides=2, padding='same', activation='selu')(out_time)
            out_time = BatchNormalization()(out_time)
            out_time = Dropout(drop_percent)(out_time)
            
            out_time = Conv2D(128,(1,3), strides=2, padding='same', activation='selu')(out_time)
            out_time = BatchNormalization()(out_time)
            out_time = Dropout(drop_percent)(out_time)
            out_time = Flatten()(out_time)
            
            ###   convlution in channel   ###
            out_cha = Conv2D(128,(32,1), 
                         kernel_initializer='TruncatedNormal',
                         bias_initializer='zeros',
                         strides=1, padding='same', activation='selu')(inp)
            out_cha = BatchNormalization()(out_cha)
            out_cha = Dropout(drop_percent)(out_cha)
            out_cha = Flatten()(out_cha)
            
            ###   2D convlution   ###
            #conv1
            out_2d = Conv2D(16,(3,3), 
                         kernel_initializer='TruncatedNormal',
                         bias_initializer='zeros',
                         strides=1, padding='same', activation='selu')(inp)
            out_2d = BatchNormalization()(out_2d)
            #out1 = out
            #out = MaxPooling2D(pool_size=(2,2), strides=2, padding='same', name = 'maxpool_1')(out)
            out_2d = Conv2D(16,(3,3), strides=1, padding='same', activation='selu')(out_2d)
            out_2d = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(out_2d)
            out_2d = BatchNormalization()(out_2d)
            out_2d = Dropout(drop_percent)(out_2d)

            #conv2
            out_2d = Conv2D(32,(3,3), strides=1, padding='same', activation='selu')(out_2d)
            out_2d = BatchNormalization()(out_2d)
            out_2d = Conv2D(32,(3,3), strides=1, padding='same', activation='selu')(out_2d)
            out_2d = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(out_2d)
            out_2d = BatchNormalization()(out_2d)
            out_2d = Dropout(drop_percent)(out_2d)

            #conv3
            out_2d = Conv2D(64,(3,3), strides=1, padding='same', activation='selu')(out_2d)
            out_2d = BatchNormalization()(out_2d)
            out_2d = Conv2D(64,(3,3), strides=1, padding='same', activation='selu')(out_2d)
            out_2d = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(out_2d)
            out_2d = BatchNormalization()(out_2d)
            out_2d = Dropout(drop_percent)(out_2d)
            out1_2d = Flatten()(out_2d)

            #conv4
            out_2d = Conv2D(128,(3,3), strides=1, padding='same', activation='selu')(out_2d)
            out_2d = BatchNormalization()(out_2d)
            out_2d = Conv2D(128,(3,3), strides=1, padding='same', activation='selu')(out_2d)
            out_2d = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(out_2d)
            out_2d = BatchNormalization()(out_2d)
            out_2d = Dropout(drop_percent)(out_2d)


            out2_2d = Flatten()(out_2d)
            out_2d = merge([out1_2d, out2_2d], mode='concat')
            #print out.shape
            
            out = merge([out_time, out_cha, out_2d], mode='concat')
            out = Dense(1024, activation='selu')(out)
            out = Dense(512, activation='selu')(out)
            out = Dense(2, activation='sigmoid')(out)

            model = Model(inp,out)

            sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)     #学习率lr的调整至关重要
            #model.compile(loss='msle', optimizer=sgd)

            label = keras.utils.to_categorical(y_train, num_classes = 2)

            test_label = keras.utils.to_categorical(y_test, num_classes = 2)

            #Another way to define your optimizer
            adam = Adam(lr=1e-5)

            #We add metrics to get more results you want see
            model.compile(optimizer=sgd,
                    loss='msle',
                    #loss='categorical_crossentropy',
                    metrics=['accuracy'])

            #print '\nNo.%d Training----------' %(v+1)       #i表示第几次交叉验证
            #Another way to train the model
            model.fit(X_train, y_train, epochs = nb_ep, 
                      batch_size= nb_bsize ,
                      verbose = print_command, shuffle=True,
                      validation_data = (X_test, y_test)
                      #validation_split=0.2, #callbacks=[save_best, early_stop],
                      )

            #model visualization
            #plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)
            #print '\nmodel graph has been saved.'

            #save model parameters
            #model.save('my_cnn_model.h5')

            #y_pred = model.predict_classes(X_test)

            print 'No.%d Testing----------' %(v+1)     
            #Evaluate the model with the metrics we defined earlier
            loss, accuracy = model.evaluate(X_test, y_test)
            
            #print 'train loss: ', loss
            #print 'train accuracy: ', accuracy

            sum_loss = sum_loss + loss
            sum_acc = sum_acc + accuracy
            
            cross_loss += loss
            cross_acc += accuracy
            #print '\nTesting----------'        
            #Evaluate the model with the metrics we defined earlier
            #loss, accuracy = model.evaluate(X_test, y_test)
            
            """
            C = model.predict(X_test)
            CSP , C
            lstm
            """
            
        #print '\naverage cross loss: ', cross_loss/Cro_v
        #print '\naverage cross accuracy: ', cross_acc/Cro_v
        
    Results_Table.add_row([sub, "p"+str(k1), sum_loss/(cross_time * Cro_v), sum_acc/(cross_time * Cro_v)]) 
    print (Results_Table)
    
#model.summary()

end_time = time.time()
sumtime = (end_time - start_time)

#print '\n---------- final loss ----------: ', sum_loss/(cross_time * Cro_v)
#print '\n---------- final accuracy ----------: ', sum_acc/(cross_time * Cro_v)
print '\n ----------- total time ----------: ', sumtime/3600, 'h'


