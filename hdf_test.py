import matplotlib.pyplot as plt
import numpy as np
import h5py

RECORDING_NR = 1

hdf5_path = './data/processed/aps_recording1_36x36.hdf5'

hdf5_f = h5py.File(hdf5_path,'r')

frames = hdf5_f['images'][-30:]
labels = x_test = hdf5_f['labels'][-30:]

for k, frame in enumerate(frames):
    plt.imshow(frame.T, cmap="gray")
    plt.savefig('./fig/' + "test_" + str(k) + ".png")

i = 0