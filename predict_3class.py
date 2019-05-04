#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
import sys
from os.path import isfile, join
# Data loading and preprocessing
#from tflearn.datasets import cifar10

import pandas as pd
from PIL import Image
import numpy as np
import sys

pic_name = pd.read_csv("pic_name.csv")["filename"].tolist()

start=121
end=140


pic_name1 = []
for i in range(len(pic_name)):
	for folder in range(start, end+1):
		if pic_name[i].startswith("%03d" % folder):
			pic_name1.append(pic_name[i])
#print(pic_name)
#print(pic_name1)
#print(len(pic_name))
#print(len(pic_name1))
pic_name=pic_name1
def find_pic_index_from_name_list(folder):
	res=[]
	for i in range(len(pic_name)):
		if pic_name[i].startswith(folder):
			res.append(i)
	#print(res)
	return res
#find_pic_index_from_name_list("002")



#x = np.load("%03d_%03d_for_test/learn_data.npy" % (start,end))
#y = np.load("%03d_%03d_for_test/learn_labels.npy" % (start,end))

x_test_total = np.load("%03d_%03d_for_test/test_data.npy" % (start,end))
y_test_total = np.load("%03d_%03d_for_test/test_labels.npy" % (start,end))

#print(len(x))
print("total test length")
print(len(y_test_total))
#print(y)


train_or_predict = str(sys.argv[1])
model_type = str(sys.argv[2])

# Convolutional network building
if model_type == '1':
        network = input_data(shape=[None, 360, 18, 3])
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 90, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.0001)
if model_type == '1_3class':
        network = input_data(shape=[None, 360, 18, 3])
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 18*3, activation='softmax')
        network = regression(network, optimizer='adam',
                                  loss='categorical_crossentropy',
                                  learning_rate=0.0001)
if model_type == '1_3class_1':
        network = input_data(shape=[None, 360, 18, 3])
        network = conv_2d(network, 32, [5,5], activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 18*3, activation='softmax')
        network = regression(network, optimizer='adam',
                               loss='categorical_crossentropy',
                               learning_rate=0.0001)
        
        
if model_type == '4':
        network = input_data(shape=[None, 180, 18, 3])
        network = conv_2d(network, 32, 2, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 2, activation='relu')
        network = conv_2d(network, 64, 2, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 1024, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 90, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.0001)


model = tflearn.DNN(network, tensorboard_verbose=0)
model.load(("%03d_%03d_for_test/model" % (start,end))+model_type+"/model.tflearn")
#result = model.predict(x_test)



def emotion_to_vec(x):
    d = np.zeros(5)
    d[x] = 1.0
    return d

def result_edit(lis):
        dic ={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[]}
        kai_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        result = []
        result1 = []
        for k in kai_list:
                dic[k]=lis[k*3:k*3+3].tolist()
                d = [0,0,0]
                #print(dic[k])
                d[(dic[k]).index(max(dic[k]))]=1.0/18
                #print(d)
                for i in d:
                        result.append(i)
        #print(result)

        for k in kai_list:
                dic[k]=lis[k*3:k*3+3].tolist()
                result1.append((dic[k]).index(max(dic[k])))
        return result1


big_dic ={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0}
#big_dic_detailed = {0:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},1:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},2:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},3:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},4:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},5:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},6:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},7:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},8:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},9:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},10:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},11:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},12:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},13:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},14:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},15:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},16:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},17:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}}

big_dic_detailed = {0:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},1:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},2:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},3:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},4:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},5:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},6:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},7:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},8:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},9:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},10:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},11:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},12:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},13:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},14:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},15:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},16:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]},17:{0:[0,0,0],1:[0,0,0],2:[0,0,0],3:[0,0,0],4:[0,0,0]}}

for folder in range(start, end+1):
        x_test = [x_test_total[index] for index in find_pic_index_from_name_list("%03d" % folder)]
        y_test = [y_test_total[index] for index in find_pic_index_from_name_list("%03d" % folder)]
        print("-------------------folder , length of x_test, y_test-------------------------")
        print(folder, len(x_test),len(y_test))
        if not len(x_test) is 0:
                result = model.predict(x_test)
        dic ={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0}
        accuracy = 0
        #dic_actual_detailed={0:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},1:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},2:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},3:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},4:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},5:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},6:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},7:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},8:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},9:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},10:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},11:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},12:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},13:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},14:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},15:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},16:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},17:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}}
        dic_actual_detailed={0:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},1:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},2:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},3:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},4:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},5:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},6:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},7:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},8:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},9:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},10:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},11:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},12:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},13:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},14:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},15:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},16:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},17:{0:[0,0,0],1:[0,0,0],2:[0,0,0]}}
        #dic_detailed = {0:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},1:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},2:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},3:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},4:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},5:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},6:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},7:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},8:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},9:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},10:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},11:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},12:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},13:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},14:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},15:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},16:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},17:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}}
        dic_detailed = {0:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},1:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},2:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},3:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},4:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},5:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},6:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},7:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},8:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},9:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},10:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},11:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},12:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},13:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},14:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},15:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},16:{0:[0,0,0],1:[0,0,0],2:[0,0,0]},17:{0:[0,0,0],1:[0,0,0],2:[0,0,0]}}
        num_zero = 0
        for i in range(len(y_test)):
                #print(find_pic_index_from_name_list("%03d" % folder))
                la=np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))
                dic[la]=dic[la]+1
                big_dic[la]=big_dic[la]+1
                if result_edit(y_test[i]) == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] and result_edit(result[i]) == result_edit(y_test[i]):
                        num_zero = num_zero + 1
                for j in range(18):
                        dic_actual_detailed[j][result_edit(y_test[i])[j]][result_edit(y_test[i])[j]] += 1
                        dic_detailed[j][result_edit(y_test[i])[j]][result_edit(result[i])[j]] += 1
                        big_dic_detailed[j][result_edit(y_test[i])[j]][result_edit(result[i])[j]] += 1
                if result_edit(result[i]) == result_edit(y_test[i]):
                        accuracy = accuracy + 1
                else:
                        k=1
                        #print("                 id:"+str(i))
                        #print(result_edit(y_test[i]))
                        #print(result_edit(result[i]))
                        #print(np.array(result_edit(result[i]))-np.array(result_edit(y_test[i])))
                        #print("same label number")
                #dic[np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))]=dic[np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))]+1
        #print(dic)
        #print(accuracy/len(y_test))
        #print(num_zero)
        #print(dic_detailed)
        #print(dic_actual_detailed)

        #---------------------------------------
        print("number by successful floor")
        for i in range(19):
                print(str(i)+","+str(dic[i]))
        print("each floor by number")
        for i in range(18):
                print("\n")
                for j in range(3):
                        print(dic_detailed[i][j])
        print("each floor by percent")
        for i in range(18):
                print("\n")
                for j in range(3):
                        sum=float(np.sum(np.array(dic_detailed[i][j])))/100
                        if sum==0:
                                print(list(np.array(dic_detailed[i][j])))
                        else:
                                print(list(np.array(dic_detailed[i][j])/float(sum)))
        print("total by number and percent")
        table_dic_for_total ={0:np.array([0,0,0]),1:np.array([0,0,0]),2:np.array([0,0,0])}
        for i in range(18):
                for j in range(3):
                        table_dic_for_total[j]=table_dic_for_total[j]+np.array(dic_detailed[i][j])
        for j in range(3):
                print(list(table_dic_for_total[j]))
        for j in range(3):
                print(list(100*table_dic_for_total[j]/float(np.sum(table_dic_for_total[j]))))
        print("total accuracy")
        matched=0
        alll=0
        for j in range(3):
                matched=matched+table_dic_for_total[j][j]
                alll=alll+np.sum(table_dic_for_total[j])
        print(matched/float(alll))
        #---------------------------------------
print("big_dic and big_dic_detailed")
print(big_dic)
print(big_dic_detailed)
