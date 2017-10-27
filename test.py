import keras
from keras.models import load_model
import h5py
import misc
import numpy as np
import time

MODEL = 'weights.30-0.38.h5'
model_path = './model/aps36_B32_60E_exp_nr/' + MODEL

TESTFILE = 'test.hdf5'
test_data_path = './data/processed/aps_36_exp_newresize/' + TESTFILE

img_rows = 36
img_cols = 36


test_h5 = h5py.File(test_data_path,'r')

x = test_h5['images'][:]
dimensions = (x.shape[0], img_rows, img_cols, 1)
x = np.reshape(x, dimensions).astype('float32')

y = test_h5['labels'][:]
y = misc.to_categorical(y, 4)


model = load_model(model_path)

start_time = time.time()
predictions = model.predict(x)
delta_t = (time.time() - start_time)/x.shape[0]
print("--- Prediction took %s seconds ---" % delta_t)

np.argmax(predictions, axis=1)

correct_predictions = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
accuracy = correct_predictions/float(y.shape[0])

print('TEST ACCURACY: ' + str(accuracy))

