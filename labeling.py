import sys
from PyAedatTools.ImportAedat import ImportAedat
import numpy as np
import matplotlib.pyplot as plt
import h5py


EVENTS_PER_FRAME = 5000
FRAME_DIM = (240,180)

inputfile = open('./data/targets/DAVIS240C-2016-01-11T15-50-11+0000-04010058-0_recording_2-targets.txt')
lines = inputfile.readlines()


timestamps = []
labels = []

for line in lines:
    line = line.strip()
    if line[0] == '#':
        continue
    timestamps.append(int(line.split(' ')[1]))
    x_coord = int(line.split(' ')[2])

    if x_coord == -1:
        labels.append(1) # N = 1
    elif x_coord < 80:
        labels.append(2) # L = 2
    elif x_coord < 160:
        labels.append(3) # C = 3
    elif x_coord < 240:
        labels.append(4) # R = 4
    else:
        labels.append(5) # Invalid = 5


# Create a dict with which to pass in the input parameters.
aedat = {}
aedat['importParams'] = {}
aedat['info'] = {}

aedat['importParams']['endEvent'] = 1e6;

#aedat['importParams']['filePath'] = './data/ball1.aedat'
aedat['importParams']['filePath'] = './data/aedat/DAVIS240C-2016-01-11T15-50-11+0000-04010058-0_recording_2.aedat'


aedat = ImportAedat(aedat)

img = np.zeros(FRAME_DIM)

filenames = [] #for gif generation

i = 0
k = 0
last_j = 0

tmp_frame_timestamps = []
NR_FRAMES = int(len(aedat['data']['polarity']['timeStamp'])/EVENTS_PER_FRAME)
frame_labels = np.zeros(NR_FRAMES+1)

f = h5py.File("./data/processed/recording2.hdf5", "w")
d_img = f.create_dataset("images", (NR_FRAMES,FRAME_DIM[0],FRAME_DIM[1]), dtype='f')
d_label = f.create_dataset("labels", (NR_FRAMES,), dtype='i')


for t,x,y,p in zip(aedat['data']['polarity']['timeStamp'], aedat['data']['polarity']['x'], aedat['data']['polarity']['y'], aedat['data']['polarity']['polarity']):
    if p==True:
        img[FRAME_DIM[0]-1-x][FRAME_DIM[1]-1-y] = 1
    else:
        img[FRAME_DIM[0]-1-x][FRAME_DIM[1]-1-y] = 0.4
    i += 1
    tmp_frame_timestamps.append(t)

    if i%EVENTS_PER_FRAME == 0:
        # plt.imshow(img.T, cmap="hot")
        # filenames.append('./fig' + "noisy_metro_" + str(k) + ".png")
        # plt.savefig('./fig/' + "noisy_metro_" + str(k) + ".png")
        # img = np.zeros((240, 180))

        for j in range(last_j,len(timestamps)):
            if timestamps[j] > tmp_frame_timestamps[-1]:
                if k>0:
                    d_label[k] = d_label[k-1]
                break

            if timestamps[j] in tmp_frame_timestamps:
                d_label[k] = labels[j] # TODO: fill the missing labels of the first few frames..
                last_j = j
                print(j,k,labels[j]) # DEBUG
                break # just take the label of the first timestamp that matches

        tmp_frame_timestamps = []
        d_img[k] = img
        k += 1

# Fill up the labels of the first frames
i=0
for label in d_label:
    if label != 0:
        d_label[:i] = label
    else:
        i += 1


# TODO: last frame with size < ...


# plt.imshow(img, cmap="hot")
# plt.colorbar()
# plt.show()