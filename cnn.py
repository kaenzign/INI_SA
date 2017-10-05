'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import h5py
import misc

batch_size = 8
num_classes = 4
epochs = 12

# input image dimensions
img_rows, img_cols = 240, 180

hdf5_train = h5py.File("./data/processed/train.hdf5",'r')
hdf5_test = h5py.File("./data/processed/test.hdf5",'r')

# x_test = hdf5_train['images'][-10:] # TODO: load seperate hdf5 file with validation data
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#
# y_test = hdf5_train['labels'][-10:]
# y_test = keras.utils.to_categorical(y_test, num_classes)

train_batches = misc.generate_batches_from_hdf5_file(hdf5_file=hdf5_train,
                                                     batch_size=batch_size,
                                                     dimensions=(batch_size,img_rows,img_cols,1),
                                                     num_classes=num_classes,
                                                     shuffle=True)

test_batches = misc.generate_batches_from_hdf5_file(hdf5_file=hdf5_test,
                                                    batch_size=batch_size,
                                                    dimensions=(batch_size,img_rows,img_cols,1),
                                                    num_classes=num_classes,
                                                    shuffle=False)


model = Sequential()
model.add(Conv2D(4, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(4, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))
#model.add(Dense(num_classes))
model.add(Dense(num_classes, activation='softmax'))
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(img_rows, img_cols, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# for e in range(epochs):
#     print("epoch %d" % e)
#     for x_batch, y_batch in train_batches:
#         model.fit(x_batch, y_batch, batch_size=32, nb_epoch=1)

num_train_batches_per_epoch = int((len(hdf5_train['labels']) - 1) / batch_size)
num_test_batches_per_epoch = int((len(hdf5_test['labels']) - 1) / batch_size)

model.fit_generator(generator=train_batches,
                    steps_per_epoch=num_train_batches_per_epoch,
                    nb_epoch=epochs,
                    validation_data=test_batches,
                    validation_steps=num_test_batches_per_epoch)


# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])