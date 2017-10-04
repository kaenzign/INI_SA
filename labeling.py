from PyAedatTools.ImportAedat import ImportAedat
import numpy as np
import matplotlib.pyplot as plt
import h5py
import misc
from tqdm import tqdm
import filenames
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-rec", "--recording", type=int, help="nunber of the recording to process")
parser.add_argument("-tag", "--tag", type=str, help="tag for hdf5 file name")
args = parser.parse_args()

print('PROCESSING RECORDING NR. ' + str(args.recording))

EULER = True
RESIZE = False
SCALE_AND_CLIP = True
EVENTS_PER_FRAME = 5000
FRAME_DIM = (240,180)
if RESIZE:
    TARGET_DIM = (72,72)
else:
    TARGET_DIM = FRAME_DIM

dim_scale = [FRAME_DIM[0]/float(TARGET_DIM[0]), FRAME_DIM[1]/float(TARGET_DIM[1])]

if EULER:
    inputfile = open('../../../scratch/kaenzign/dvs_targets/' + filenames.target_names[args.recording-1])
else:
    inputfile = open('./data/targets/' + filenames.target_names[args.recording - 1])

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

# aedat['importParams']['endEvent'] = 3e5;

#aedat['importParams']['filePath'] = './data/ball1.aedat'

if EULER:
    aedat['importParams']['filePath'] = '../../../scratch/kaenzign/aedat/' + filenames.aedat_names[args.recording-1]
else:
    aedat['importParams']['filePath'] = './data/aedat/' + filenames.aedat_names[args.recording - 1]



aedat = ImportAedat(aedat)

#img = np.zeros(FRAME_DIM)
img = np.full(TARGET_DIM, 0.5)


filenames = [] #for gif generation

i = 0
k = 0
last_j = 0

tmp_frame_timestamps = []
NR_FRAMES = int(len(aedat['data']['polarity']['timeStamp'])/EVENTS_PER_FRAME)
frame_labels = np.zeros(NR_FRAMES+1)

if EULER:
    hdf5_name = '../../../scratch/kaenzign/processed/dvs_recording' + str(args.recording)
else:
    hdf5_name = './data/processed/dvs_recording' + str(args.recording)
# hdf5_name += '_' + str(int(time.time()))
if args.tag:
    hdf5_name += '_' + args.tag
hdf5_name += '.hdf5'


f = h5py.File(hdf5_name, "w")
d_img = f.create_dataset("images", (NR_FRAMES,TARGET_DIM[0],TARGET_DIM[1]), dtype='f')
d_label = f.create_dataset("labels", (NR_FRAMES,), dtype='i')


for t,x,y,p in tqdm(zip(aedat['data']['polarity']['timeStamp'], aedat['data']['polarity']['x'], aedat['data']['polarity']['y'], aedat['data']['polarity']['polarity'])):
    if RESIZE:
        x = int(x/dim_scale[0])
        y = int(y/dim_scale[1])

    if p==True:
        img[TARGET_DIM[0]-1-x][TARGET_DIM[1]-1-y] += 0.005
        #img[TARGET_DIM[0]-1-x][TARGET_DIM[1]-1-y] += 1
    else:
        img[TARGET_DIM[0]-1-x][TARGET_DIM[1]-1-y] -= 0.005
        # img[TARGET_DIM[0]-1-x][TARGET_DIM[1]-1-y] += 1

    tmp_frame_timestamps.append(t)

    i += 1

    if i%EVENTS_PER_FRAME == 0:
        if SCALE_AND_CLIP:
            img = misc.three_sigma_frame_clipping(img)
            img = misc.frame_scaling(img)

        # plt.imshow(img.T, cmap="gray")
        # filenames.append('./fig' + "noisy_metro_" + str(k) + ".png")
        # plt.savefig('./fig/' + "noisy_metro_" + str(k) + ".png")

        for j in range(last_j,len(timestamps)):
            if timestamps[j] > tmp_frame_timestamps[-1]:
                if k>0:
                    d_label[k] = d_label[k-1]
                break

            if timestamps[j] in tmp_frame_timestamps:
                d_label[k] = labels[j]
                last_j = j
                # print(j,k,labels[j]) # DEBUG
                break # just take the label of the first timestamp that matches

        tmp_frame_timestamps = []
        d_img[k] = img
        img = np.full(TARGET_DIM, 0.5)
        k += 1

# Fill up the labels of the first frames
i=0
for label in d_label:
    if label != 0:
        d_label[:i] = label
    else:
        i += 1



# plt.imshow(img, cmap="hot")
# plt.colorbar()
# plt.show()