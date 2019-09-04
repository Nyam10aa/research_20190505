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
def read_from_s3(file_path, file_name):
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket='takenaka', Key = file_path + file_name)
        return io.BytesIO(obj['Body'].read())
case_type = int(sys.argv[1])

def case_2():
	#train_list = list(range(36,141))
	#train_list = [3, 56, 88, 106, 116, 145, 172, 244, 245, 309, 445, 500, 503, 506, 563, 568, 575, 587, 612, 632, 633, 634, 636, 655, 691, 712]+[8, 57, 61, 156, 164, 197, 254, 269, 286, 297, 340, 343, 352, 408, 417, 430, 448, 457, 462, 482, 524, 525, 588, 595, 616, 617, 640, 665, 675, 677, 679, 689, 711]# + list(range(106,141))
        train_list = [1,2,3,6,8,9,18,21,22,23,24,25,26,27,30,31,32,35,37,38,39,43,44,45,49,51,52,54,57,60,62,63,64,69,72,73,75,79,80,81,82,84,85,86,87,90,93,96,102,104,105,106,107,108,109,110,112,113,115,116,118,119,120,123,124,125,126,127,128,129,133,136,137,139,140,141,143,144,148,149,150,153,154,157,167,169,170,171,172,173,176,177,179,180,194,197,200,202,203,204,207,213,219,225,226,231,237,239,242,244,245,246,247,250,253,255,262,269,274,275,282,283,286,289,291,292,299,300,301,302,303,304,308,310,312,313,316,318,319,320,322,326,336,337,341,342,351,352,353,354,358,360,361,363,365,367,371,376,377,382,384,385,386,401,402,404,405,406,407,408,410,411,412,415,416,417,418,420,421,422,423,425,426,427,428,429,430,434,438,440,441,442,443,444,445,446,447,448,449,451,452,453,455,456,457,458,462,463,465,469,480,482,483,484,486,488,489,490,492,493,494,497,499,500,501,502,504,505,506,507,508,509,510,512,513,514,515,517,518,523,524,527,528,529,531,536,537,538,542,543,544,549,553,559,563,564,565,567,568,569,570,573,574,575,576,577,578,579,580,581,582,583,584,586,587,588,589,590,591,592,593,594,596,597,598,599,600,601,603,604,605,609,611,613,614,616,617,618,619,623,624,628,630,631,632,633,635,636,638,642,643,644,645,647,648,649,651,652,654,655,657,658,659,660,661,662,663,667,668,669,670,675,676,677,679,680,681,682,684,685,688,689,690,691,692,693,695,696,697,698,699,700,701,702,703,704,705,706,711,714,715,716,717,720,721,722,723]
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
hist=model2.fit(x,y, validation_split=0.01, batch_size=300, epochs=15)#100) 
print(hist.history)
model2.save('divided_by_20_keras_model/M2_kaigi5/model_case_%s.h5' % (case_type), include_optimizer=False)

