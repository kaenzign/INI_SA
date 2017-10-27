from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import h5py
import misc
#import matplotlib.pyplot as plt
import json
import os


MODEL_TAG = 'dvs_36_batchN_avgPool'
EULER = False
TENSORBOARD = False
CHECKPOINTS = True

batch_size = 32
num_classes = 4
epochs = 10

# input image dimensions
img_rows, img_cols = 36, 36

if EULER:
    processed_path = '../../../scratch/kaenzign/processed/'
else:
    processed_path = './data/processed/'

processed_path += 'aps_36_exp_newresize/'

hdf5_train = h5py.File(processed_path + 'train.hdf5','r')
hdf5_test = h5py.File(processed_path + 'test.hdf5','r')

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
model.add(Conv2D(4, kernel_size=(5, 5), input_shape=(img_rows, img_cols, 1), bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(4, (5, 5), bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(40, bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(num_classes, bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('softmax'))

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

if MODEL_TAG != '':
    log_dir = './log/' + MODEL_TAG + '/'
    model_dir = './model/'+ MODEL_TAG + '/'
else:
    log_dir = './log'
    model_dir = './model/'

if TENSORBOARD:
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                     write_graph=True,
                     write_images=False)
    callbacks = [tensorboard_cb]
else:
    callbacks = None
# tensorboard --logdir=./logs

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if CHECKPOINTS:
    model_checkpoints = keras.callbacks.ModelCheckpoint(model_dir + 'weights.{epoch:02d}-{val_loss:.2f}.h5')
    if callbacks == None:
        callbacks = [model_checkpoints]
    else:
        callbacks.append(model_checkpoints)

history = model.fit_generator(generator=train_batches,
                    steps_per_epoch=num_train_batches_per_epoch,
                    nb_epoch=epochs,
                    validation_data=test_batches,
                    validation_steps=num_test_batches_per_epoch,
                    callbacks=callbacks)


if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(model_dir + 'mdl.h5')

# list all data in history
print(history.history.keys())

with open(model_dir + 'history.json', 'w') as f:
    json.dump(history.history, f)

# elsewhere...

# with open('./model/my_dict.json') as f:
#     my_dict = json.load(f)

# summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])