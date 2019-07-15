from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam

import keras
import numpy as np
import sys
import boto3
import io

#x = np.load("l%03d_%03d_for_test/learn_data.npy" % (start,end))
#y = np.load("l%03d_%03d_for_test/learn_labels.npy" % (start,end))

#x_test = np.load("l%03d_%03d_for_test/test_data.npy" % (start,end))
#y_test = np.load("l%03d_%03d_for_test/test_labels.npy" % (start,end))


#x = np.load("divided_by_20/data_001_020.npy")
#y = np.load("divided_by_20/labels_001_020.npy")
#x_test = np.load("divided_by_20/data_121_140.npy")
#y_test = np.load("divided_by_20/labels_121_140.npy")
case_type = int(sys.argv[1])

def case_2():
	#train_list = list(range(36,141))
	train_list = [8, 57, 61, 156, 164, 197, 254, 269, 286, 297, 340, 343, 352, 408, 417, 430, 448, 457, 462, 482, 524, 525, 588, 595, 616, 617, 640, 665, 675, 677, 679, 689, 711]# + list(range(106,141))
	print("train set")
	print(train_list)
	case = train_list[0]
	x = np.load(read_from_s3('processed_data_STD/','datas_%03d_%03d.npy' % (case, case)))
	y = np.load(read_from_s3('processed_data_STD/','labels_%03d_%03d.npy' % (case, case)))
	for case in train_list[1:]:
		print("check")
		print(len(x))
		print(len(y))
		x1 = np.load(read_from_s3('processed_data_STD/','datas_%03d_%03d.npy' % (case, case)))
		y1 = np.load(read_from_s3('processed_data_STD/','labels_%03d_%03d.npy' % (case, case)))
		x = np.concatenate([x, x1])
		y = np.concatenate([y, y1])
	return x, y

if case_type == 2:
	x,y = case_2()

def read_from_s3(file_path, file_name):
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket='takenaka', Key = file_path + file_name)
        return io.BytesIO(obj['Body'].read())

#x = np.load("./divided_by_20/datas_%03d_%03d.npy" % (case_type, case_type))
#y = np.load("./divided_by_20/labels_%03d_%03d.npy" % (case_type, case_type))
#x = np.load(read_from_s3('processed_data_STD/','datas_%03d_%03d.npy' % (case_type, case_type)))
#y = np.load(read_from_s3('processed_data_STD/','labels_%03d_%03d.npy' % (case_type, case_type)))



print(x.shape)
print(y.shape)
#X = x.reshape(x.shape[0], 360*18*3)
#X_test = x_test.reshape(x_test.shape[0], 360*18*3)
#print(X.shape)

#---------------------------------------------------------------------------------
model = Sequential()
model.add(Dense(30, input_dim=360*18*3, activation='relu'))
model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(18*3, activation='softmax'))
#--------------------------------------------------------------------------------
model2 = Sequential()
model2.add(Conv2D(32,(20, 5), padding='valid', activation='relu', input_shape=(360, 18, 1)))
#model2.add(Dropout(0.5))
#model2.add(BatchNormalization())
#model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(32, (20,3), padding='valid', activation='relu'))
model2.add(Conv2D(32, (20,3), padding='valid', activation='relu'))
model2.add(Conv2D(32, (20,3), padding='valid', activation='relu'))
model2.add(Conv2D(32, (20,3), padding='valid', activation='relu'))
model2.add(Conv2D(32, (20,3), padding='valid', activation='relu'))
#model2.add(Conv2D(32, 5, padding='valid', activation='relu'))
#model2.add(MaxPooling2D(pool_size=(2, 2)))

#model2.add(Dropout(0.5))
#model2.add(BatchNormalization())
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
#model2.add(Dropout(0.5))
model2.add(Dense(18*3, activation='softmax'))

model2.summary()                             
model2.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


#num_epoch=20
#model.fit(X, y, validation_data=(X_test, y_test), callbacks=[TestCallback((X_test, y_test))], epochs=num_epoch+1, batch_size=100,verbose=0)
#for i in range(10):
hist=model2.fit(x,y, validation_split=0.01, batch_size=100, epochs=100) 
print(hist.history)
model2.save('divided_by_20_keras_model/M2_kaigi4/model_case_%s.h5' % (case_type), include_optimizer=False)

