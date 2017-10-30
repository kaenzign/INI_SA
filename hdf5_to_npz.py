import numpy as np
import h5py
import misc

NR_SAMPLES = 500
TESTFILE = 'test_dvs.hdf5'
TRAINFILE = 'train_dvs.hdf5'
test_data_path = './data/processed/' + TESTFILE
train_data_path = './data/processed/' + TRAINFILE


test_h5 = h5py.File(test_data_path,'r')
train_h5 = h5py.File(test_data_path,'r')

random_indexes = np.random.randint(0, test_h5['images'].shape[0], size=NR_SAMPLES)

x_test = np.array(test_h5['images'])[random_indexes]
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_test = np.array(test_h5['labels'])[random_indexes]
y_test = misc.to_categorical(y_test)

x_norm = test_h5['images'][:]

np.savez(file='x_test', arr_0=x_test)
np.savez(file='y_test', arr_0=y_test)

np.savez(file='x_norm', arr_0=x_norm)

y_t = np.load('y_test.npz')
y_t = y_t['arr_0']

x_t = np.load('x_test.npz')
x_t = x_t['arr_0']

i=0