from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.constraints import non_neg
from keras import regularizers
import h5py
import misc
#import matplotlib.pyplot as plt
import json
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", type=int, help="nunber of neurons")
parser.add_argument("-N", "--neurons", type=int, help="nunber of neurons")
parser.add_argument("-tag", "--tag", type=str, help="tag for hdf5 file name")
args = parser.parse_args()

if args.tag!=None:
    MODEL_TAG = args.tag
else:
    MODEL_TAG = 'D8_B'
EULER = False
TENSORBOARD = False
CHECKPOINTS = True
USE_BIAS = True
BIAS_REGULARIZER = None
BATCH_NORMALIZATION = True
WEIGHT_CONSTRAINT = non_neg()
WEIGHT_CONSTRAINT = None
if args.neurons != None:
    NEURONS = args.neurons
else:
    NEURONS = 8
if args.model != None:
    MODEL = args.model
else:
    MODEL = 2 # 1: CNN, 2: MLP

batch_size = 32
num_classes = 4
epochs = 30

# input image dimensions
img_rows, img_cols = 36, 36

if EULER:
    processed_path = '../../../scratch/kaenzign/processed/'
else:
    processed_path = './data/processed/'

processed_path += 'dvs_36_evtacc_corrected/'

hdf5_train = h5py.File(processed_path + 'train.hdf5','r')
hdf5_test = h5py.File(processed_path + 'test.hdf5','r')

dimensions = (batch_size,img_rows,img_cols,1)
#dimensions = (batch_size,img_rows*img_cols)


train_batches = misc.generate_batches_from_hdf5_file(hdf5_file=hdf5_train,
                                                     batch_size=batch_size,
                                                     dimensions=dimensions,
                                                     num_classes=num_classes,
                                                     shuffle=True)

test_batches = misc.generate_batches_from_hdf5_file(hdf5_file=hdf5_test,
                                                    batch_size=batch_size,
                                                    dimensions=dimensions,
                                                    num_classes=num_classes,
                                                    shuffle=False)

if MODEL==1:
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(5, 5), input_shape=(img_rows, img_cols, 1), bias_regularizer=BIAS_REGULARIZER, use_bias=USE_BIAS))
    # if BATCH_NORMALIZATION:
    #     model.add(BatchNormalization())
    model.add(Activation('relu'))

    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(4, (5, 5), bias_regularizer=BIAS_REGULARIZER, use_bias=USE_BIAS))
    # if BATCH_NORMALIZATION:
    #     model.add(BatchNormalization())
    model.add(Activation('relu'))

    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(40, bias_regularizer=BIAS_REGULARIZER, use_bias=USE_BIAS))
    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes, bias_regularizer=BIAS_REGULARIZER, use_bias=USE_BIAS))
    # if BATCH_NORMALIZATION:
    #     model.add(BatchNormalization())
    model.add(Activation('softmax'))

if MODEL==2:
    model = Sequential()
    model.add(Flatten(input_shape=(img_rows, img_cols, 1)))
    model.add(Dense(NEURONS, activation='relu',  bias_regularizer=BIAS_REGULARIZER, use_bias=USE_BIAS, kernel_constraint=WEIGHT_CONSTRAINT))
    #model.add(Dense(16, activation='relu', input_shape=(img_cols*img_rows,), bias_regularizer=BIAS_REGULARIZER, use_bias=USE_BIAS))
    model.add(Dropout(0.2))
    # model.add(Dense(64, activation='relu', bias_regularizer=BIAS_REGULARIZER, use_bias=USE_BIAS))
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax', bias_regularizer=BIAS_REGULARIZER, use_bias=USE_BIAS, kernel_constraint=WEIGHT_CONSTRAINT))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


num_train_batches_per_epoch = int((len(hdf5_train['labels']) - 1) / batch_size)
num_test_batches_per_epoch = int((len(hdf5_test['labels']) - 1) / batch_size)

if MODEL_TAG != '':
    log_dir = './log/' + MODEL_TAG + '/'
    model_dir = './model/new/'+ MODEL_TAG + '/'
else:
    log_dir = './log'
    model_dir = './model/new/'

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
