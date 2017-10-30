import numpy as np
import h5py
import misc
from random import shuffle

def balance_classes(x,y):
    counts = [0,0,0,0]

    indexes = {}
    indexes['N'] = []
    indexes['L'] = []
    indexes['C'] = []
    indexes['R'] = []

    for i, label in enumerate(y):
        if label == 1:
            counts[0] += 1
            indexes['N'].append(i)
        elif label == 2:
            counts[1] += 1
            indexes['L'].append(i)
        elif label == 3:
            counts[2] += 1
            indexes['C'].append(i)
        elif label == 4:
            counts[3] += 1
            indexes['R'].append(i)

    min_class_count = min(counts)
    max_class_count = max(counts)
    print('class {} has minimal count {}'.format(counts.index(min_class_count), min_class_count))
    print('class {} has minimal count {}'.format(counts.index(max_class_count), max_class_count))

    idx = indexes['N'][:min_class_count] + indexes['L'][:min_class_count] + indexes['C'][:min_class_count] + indexes['R'][:min_class_count]
    shuffle(idx)

    return x[idx], y[idx]


NR_SAMPLES = 2000
BALANCE_CLASSES = True
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

if BALANCE_CLASSES:
    x_test, y_test = balance_classes(x_test, y_test)


y_test = misc.to_categorical(y_test)


x_norm = test_h5['images'][:]
x_norm = x_norm.reshape(x_norm.shape[0], x_norm.shape[1], x_norm.shape[2], 1)

np.savez(file='x_test', arr_0=x_test)
np.savez(file='y_test', arr_0=y_test)

np.savez(file='x_norm', arr_0=x_norm)

# y_t = np.load('y_test.npz')
# y_t = y_t['arr_0']
#
# x_t = np.load('x_test.npz')
# x_t = x_t['arr_0']

i=0