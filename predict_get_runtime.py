from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam

import keras
import numpy as np
import pandas as pd
import sys
import boto3
import io
import time


start = time.time()

def read_from_s3(file_path, file_name):
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket='takenaka', Key = file_path + file_name)
        return io.BytesIO(obj['Body'].read())


# models = []
# for model_type in range(1, 724):
#     model = keras.models.load_model("divided_by_20_keras_model/by_one/model_case_%d.h5" % model_type, compile=False)
#     model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
#     models.append(model)
#     print("="*10, model_type, "="*10)
#     elapsed_time = time.time() - start
#     print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")


x_datas = []
y_datas = []
for start in range(1, 724):
    x_test_total = np.load(read_from_s3('processed_data_STD/', 'datas_%03d_%03d.npy' % (start, start)))
    y_test_total = np.load(read_from_s3('processed_data_STD/', 'labels_%03d_%03d.npy' % (start, start)))
    x_datas.append(x_test_total)
    y_test_total.append(y_test_total)
    print("="*10, start, "="*10)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")