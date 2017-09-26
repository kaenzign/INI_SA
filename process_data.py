import sys
from PyAedatTools.ImportAedat import ImportAedat
import numpy as np
import matplotlib.pyplot as plt

EVENTS_PER_FRAME = 5000
FRAME_DIM = [240,180]

# Create a dict with which to pass in the input parameters.
aedat = {}
aedat['importParams'] = {}
aedat['info'] = {}

#aedat['importParams']['endEvent'] = 2e6;

aedat['importParams']['filePath'] = './data/ball1.aedat'
aedat['importParams']['filePath'] = './data/DAVIS240C-2016-01-11T15-50-11+0000-04010058-0_recording_2.aedat'


aedat = ImportAedat(aedat)

img = np.zeros((240,180))

filenames = [] #for gif generation

i = 0
k = 0
for t,x,y,p in zip(aedat['data']['polarity']['timeStamp'], aedat['data']['polarity']['x'], aedat['data']['polarity']['y'], aedat['data']['polarity']['polarity']):
    if p==True:
        img[FRAME_DIM[0]-1-x][FRAME_DIM[1]-1-y] = 1
    else:
        img[FRAME_DIM[0]-1-x][FRAME_DIM[1]-1-y] = 0.4
    i += 1
    if i%EVENTS_PER_FRAME == 0:
        plt.imshow(img.T, cmap="hot")
        filenames.append('./fig' + "noisy_metro_" + str(k) + ".png")
        plt.savefig('./fig/' + "noisy_metro_" + str(k) + ".png")
        img = np.zeros((240, 180))
        k += 1

# TODO: last frame with size < ...


plt.imshow(img, cmap="hot")
# plt.colorbar()
plt.show()