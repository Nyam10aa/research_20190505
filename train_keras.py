from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam

import keras
import numpy as np

start=121
end=140
x = np.load("l%03d_%03d_for_test/learn_data.npy" % (start,end))
y = np.load("l%03d_%03d_for_test/learn_labels.npy" % (start,end))

x_test = np.load("l%03d_%03d_for_test/test_data.npy" % (start,end))
y_test = np.load("l%03d_%03d_for_test/test_labels.npy" % (start,end))


print(x.shape)
#X = x.reshape(x.shape[0], 360*18*3)
#X_test = x_test.reshape(x_test.shape[0], 360*18*3)
#print(X.shape)

#---------------------------------------------------------------------------------
# まずは，全結合2層だけで，分類してみます．
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
model2.add(Flatten()) # Keras の場合，conv から dense に移行する時には，テンソルをベクトルに変換する操作(Flatten)が必要です．
model2.add(Dense(512, activation='relu'))
#model2.add(Dropout(0.5))
model2.add(Dense(18*3, activation='softmax'))


#model2.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

#model2.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
model2.summary()                             
model2.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


num_epoch=20
#model.fit(X, y, validation_data=(X_test, y_test), callbacks=[TestCallback((X_test, y_test))], epochs=num_epoch+1, batch_size=100,verbose=0)
#for i in range(10):
hist=model2.fit(x,y, validation_split=0.05, batch_size=100, epochs=40) 
print(hist.history)
model2.save('keras_model/model_new.h5', include_optimizer=False)

#model = keras.models.load_model('keras_model/model.h5', compile=False)
#model.compile(loss="categorical_crossentropy",optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

#score = model.evaluate(x_test, y_test, verbose=0)
#result = model.predict(x_test)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
