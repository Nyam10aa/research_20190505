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
'''
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y)
Y_test = to_categorical(Y_test)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
'''

start=121
end=140
x = np.load("%03d_%03d_for_test/learn_data.npy" % (start,end))
y = np.load("%03d_%03d_for_test/learn_labels.npy" % (start,end))

x_test = np.load("%03d_%03d_for_test/test_data.npy" % (start,end))
y_test = np.load("%03d_%03d_for_test/test_labels.npy" % (start,end))
#x_test = np.load("test_20180108.npy")
#y_test = np.load("test_labels_20180108.npy")

print(len(x))
print(len(y))
print(y)



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
        
if model_type == '2':
	network = input_data(shape = [None, 180, 18, 3])
	network = conv_2d(network, 64, 5, activation = 'relu')
	network = max_pool_2d(network, 3, strides = 2)
	network = conv_2d(network, 64, 5, activation = 'relu')
	network = max_pool_2d(network, 3, strides = 2)
	network = conv_2d(network, 128, 4, activation = 'relu')
	network = dropout(network, 0.3)
	network = fully_connected(network, 3072, activation = 'relu')
	network = fully_connected(network, 90, activation = 'softmax')
	network = regression(network, optimizer = 'momentum', loss = 'categorical_crossentropy',learning_rate=0.0001)

if model_type == '2_new':
	network = input_data(shape = [None, 180, 18, 3])
	network = conv_2d(network, 64, 6, activation = 'relu')	#4
	network = max_pool_2d(network, 3)
	network = conv_2d(network, 64, 4, activation = 'relu')	#2
	network = max_pool_2d(network, 3)
	network = conv_2d(network, 128, 2, activation = 'relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 512, activation = 'relu')
	network = fully_connected(network, 90, activation = 'softmax')
	network = regression(network, optimizer = 'momentum', loss = 'categorical_crossentropy',learning_rate=0.0001)



if model_type == '3':
	network = input_data(shape=[None, 180, 18, 3])
	network = fully_connected(network, 4096, activation='relu')
	network = fully_connected(network, 90, activation='softmax')
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


# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)

table_dic ={
0:[0,0,0,0,0],
1:[0,0,0,0,0],
2:[0,0,0,0,0],
3:[0,0,0,0,0],
4:[0,0,0,0,0],
} 


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
		dic[k]=lis[k*5:k*5+5].tolist()
		d = [0,0,0,0,0]
		#print(dic[k])
		d[(dic[k]).index(max(dic[k]))]=1.0/18
		#print(d)
		for i in d:
			result.append(i)
	#print(result)

	for k in kai_list:
		dic[k]=lis[k*5:k*5+5].tolist()
		result1.append((dic[k]).index(max(dic[k])))
	return result1



if train_or_predict == "train":
	if isfile(("%03d_%03d_for_test/model" % (start,end))+model_type+"/model.tflearn.meta"):#train existing model
		print("training existing model")
		model.load(("%03d_%03d_for_test/model" % (start,end))+model_type+"/model.tflearn")
	model.fit(x, y, n_epoch=30, shuffle=True, validation_set=0.05,show_metric=True, batch_size=500, run_id='cifar10_cnn')
	model.save(("%03d_%03d_for_test/model" % (start,end))+model_type+"/model.tflearn")
elif train_or_predict == "predict":
	dic ={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0}
	accuracy = 0
	model.load(("%03d_%03d_for_test/model" % (start,end))+model_type+"/model.tflearn")
	print("okkkkko")
	#print(model.predict(x_test[0:10000]))
	result = model.predict(x_test)
	print("okkkkko")
	#{floor:{real label:[predict label]}}
	dic_actual_detailed={0:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},1:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},2:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},3:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},4:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},5:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},6:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},7:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},8:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},9:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},10:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},11:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},12:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},13:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},14:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},15:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},16:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},17:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}}
	dic_detailed = {0:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},1:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},2:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},3:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},4:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},5:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},6:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},7:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},8:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},9:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},10:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},11:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},12:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},13:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},14:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},15:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},16:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]},17:{0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,0,0,0,0],4:[0,0,0,0,0]}}
	num_zero = 0
	for i in range(len(y_test)):
		#print(i)
		print(i,len(y_test))
		la=np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))
		dic[la]=dic[la]+1
		print(result_edit(y_test[i]))
		print(len(result_edit(y_test[i])))
		if result_edit(y_test[i]) == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] and result_edit(result[i]) == result_edit(y_test[i]):
			num_zero = num_zero + 1
		for j in range(18):
			dic_actual_detailed[j][result_edit(y_test[i])[j]][result_edit(y_test[i])[j]] += 1
			dic_detailed[j][result_edit(y_test[i])[j]][result_edit(result[i])[j]] += 1
		if result_edit(result[i]) == result_edit(y_test[i]):
			accuracy = accuracy + 1
		else:
			k=1
			print("			id:"+str(i))
			print(result_edit(y_test[i]))
			print(result_edit(result[i]))
			print(np.array(result_edit(result[i]))-np.array(result_edit(y_test[i])))
			print("same label number")		
		#dic[np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))]=dic[np.sum(np.array(result_edit(result[i])) == np.array(result_edit(y_test[i])))]+1
	print(dic)
	print(accuracy/len(y_test))
	print(num_zero)
	print(dic_detailed)
	print(dic_actual_detailed)
	#print(result_edit(result[0]))
	#print(y_test[0])
	#print(len(y_test))
#		print i
	#	if np.argmax(result[i])==np.argmax(y_test[i]):
	#		accuracy = accuracy + 1
	#	if np.argmax(result[i]) == 0:
	#		print("haha")
	#	table_dic[np.argmax(y_test[i])][np.argmax(result[i])]+=1
	#accuracy = accuracy/len(result)
	#print("accuracy is "+str(accuracy))
	#print(table_dic)
#else:
#	print("it is not train or predict.")
else:
	print("it is not train or predict.")
