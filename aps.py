import filenames
import argparse
import numpy as np
import h5py
import misc
import imageio
import matplotlib.pyplot as plt


EULER = False
RESIZE = False

FRAME_DIM = (240,180)
if RESIZE:
    TARGET_DIM = (72,72)
else:
    TARGET_DIM = FRAME_DIM

parser = argparse.ArgumentParser()
parser.add_argument("-rec", "--recording", type=int, help="nunber of the recording to process")
parser.add_argument("-tag", "--tag", type=str, help="tag for hdf5 file name")
args = parser.parse_args()

if EULER:
    f_targets = open('../../../scratch/kaenzign/dvs_targets/' + filenames.target_names[args.recording-1])
else:
    f_targets = open('./data/targets/' + filenames.target_names[args.recording - 1])

f_aps_timecodes = open('./data/aps_timecodes/' + 'DAVIS240C-2016-01-11T15-43-32+0000-04010058-0_recording_1_APS-timecode.txt')
target_lines = f_targets.readlines()


target_timestamps = []
labels = []

for line in target_lines:
    line = line.strip()
    if line[0] == '#':
        continue
    target_timestamps.append(int(line.split(' ')[1]))
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

aps_lines = f_aps_timecodes.readlines()

aps_timecodes = []

for line in aps_lines:
    line = line.strip()
    if line[0] == '#':
        continue
    aps_timecodes.append(int(line.split(' ')[1]))

if EULER:
    hdf5_name = '../../../scratch/kaenzign/processed/aps_recording' + str(args.recording)
else:
    hdf5_name = './data/processed/aps_recording' + str(args.recording)
# hdf5_name += '_' + str(int(time.time()))
if args.tag:
    hdf5_name += '_' + args.tag
hdf5_name += '.hdf5'

f = h5py.File(hdf5_name, "w")
NR_FRAMES = len(aps_timecodes)
d_img = f.create_dataset("images", (NR_FRAMES,TARGET_DIM[0],TARGET_DIM[1]), dtype='f')
d_label = f.create_dataset("labels", (NR_FRAMES,), dtype='i')

i = 0
k = 0
for k, t_aps in enumerate(aps_timecodes):
    while t_aps > target_timestamps[i]:
        i += 1

    d_label[k] = labels[i]
    # print(k, t_aps, target_timestamps[i], aps_labels[k])


avi_filename = './data/aps_avi/' + 'DAVIS240C-2016-01-11T15-43-32+0000-04010058-0_recording_1_APS.avi'
# aps_frames = misc.avi_to_frame_list(avi_filename, gray=True)
# aps_frames[0][0][0]

vid = imageio.get_reader(avi_filename,  'ffmpeg')

for k, image in enumerate(vid.iter_data()):
    #d_img[k] = image
    d_img[k] = misc.frame_scaling(np.moveaxis(image, 2, 0)[0].T) # RGB image shape (240,180,3) --> (3,240,180) ---> [0] (240,180)
    # print(image.mean())

#plt.imshow(d_img[k].T, cmap='gray')

i=0




