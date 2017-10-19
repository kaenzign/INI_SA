import numpy as np
import h5py
import misc

NR_SAMPLES = 200
TESTFILE = 'test_aps.hdf5'
test_data_path = './data/processed/aps_36_exp_newresize/' + TESTFILE

test_h5 = h5py.File(test_data_path,'r')

random_indexes = sorted(np.random.randint(0, test_h5['images'].shape[0], size=NR_SAMPLES))

x_test = np.array(test_h5['images'])[random_indexes]
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_test = np.array(test_h5['labels'])[random_indexes]
y_test = misc.to_categorical(y_test)

np.savez(file='x_test', arr_0=x_test)
np.savez(file='y_test', arr_0=y_test)

y_t = np.load('y_test.npz')
y_t = y_t['arr_0']

x_t = np.load('x_test.npz')
x_t = x_t['arr_0']



i=0