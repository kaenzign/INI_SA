import os
import struct
import time
import argparse
import filenames
import h5py
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-rec", "--recording", type=int, help="nunber of the recording to process")
args = parser.parse_args()



print('PROCESSING RECORDING NR. ' + str(args.recording))

EULER = False
EVENTS_PER_FRAME = 5000
TEST_FRACTION  = 0.2
NR_FRAME_DIV = None # set to None to extract all frames, set to value in (0,1) to select the fraction of the test set to be used
EVT_DVS = 0  # DVS event type
EVT_APS = 1 # APS event

if EULER:
    aedat_file = '../../../scratch/kaenzign/aedat/' + filenames.aedat_names[args.recording-1]
    hdf5_name = '../../../scratch/kaenzign/processed/full_36/dvs_recording' + str(args.recording) + '_36x36.hdf5'
    full_target_file = '../../../scratch/kaenzign/aedat/full_dvs_' + str(args.recording) + '.aedat'
    test_target_file = '../../../scratch/kaenzign/aedat/test_dvs_' + str(args.recording) + '.aedat'
else:
    aedat_file = './data/aedat/' + filenames.aedat_names[args.recording - 1]
    hdf5_name = './data/processed/dvs_recording' + str(args.recording) + '_36x36.hdf5'
    full_target_file = './data/aedat/full_dvs_' + str(args.recording) + '.aedat'
    test_target_file = './data/aedat/test_dvs_' + str(args.recording) + '.aedat'



# Data format is int32 address, int32 timestamp (8 bytes total)
readMode = '>II' # >: big endian, I: unsigned int (4byte)
aeLen = 8



def parse_header(file):
    # HEADER
    p = 0 # file pointer
    header_lines = []
    lt = file.readline()
    while lt != "" and chr(lt[0]) == '#':
        header_lines.append(lt)
        p += len(lt)
        lt = file.readline()
        # print(str(lt))
    return p, header_lines

def write_test_aedat(aedat_file, target_file, fraction):

    aerdata_fh = open(aedat_file, 'rb')
    target_fh = open(target_file, 'wb')

    statinfo = os.stat(aedat_file)
    file_size = statinfo.st_size
    print ("file size", file_size)

    # HEADER
    p, header_lines = parse_header(aerdata_fh)

    target_fh.writelines(header_lines)
    header_size = p
    aerdata_fh.seek(p) # necessary as we've read one line too much in last while iteration of parse_header()

    # EVENTS
    data_size = file_size - header_size

    test_data_size = fraction*data_size
    test_data_size = int(test_data_size - (test_data_size % 8))

    nr_events = test_data_size/8

    p = file_size - test_data_size
    aerdata_fh.seek(p)

    test_data = aerdata_fh.read(test_data_size)
    target_fh.write(test_data)

    return nr_events


def check_target(aedat_file, target_file):
    aerdata_fh = open(aedat_file, 'rb')
    target_fh = open(target_file, 'rb')

    i = 0
    while True:
        # if i < 1527:
        #     orig_line = aerdata_fh.readline()
        #     target_line = target_fh.readline()
        #     i += 1
        #     continue
        orig_line = aerdata_fh.readline()
        target_line = target_fh.readline()
        if target_line != orig_line:
            print(str(i) + ' not EQ')
        i += 1

def extract_DVS_events(aedat_file, target_file):
    aerdata_fh = open(aedat_file, 'rb')
    target_fh = open(target_file, 'wb')

    statinfo = os.stat(aedat_file)
    file_size = statinfo.st_size
    print("file size", file_size)

    # HEADER
    p, header_lines = parse_header(aerdata_fh)

    target_fh.writelines(header_lines)
    header_size = p
    aerdata_fh.seek(p)  # necessary as we've read one line too much in last while iteration

    # EVENTS
    s = aerdata_fh.read(aeLen) # read the first 8 byte
    p += aeLen
    while p < file_size:
        addr, ts = struct.unpack(readMode, s)

        # parse event type
        eventtype = (addr >> 31)

        if eventtype == EVT_DVS:
            target_fh.write(s)
        # if eventtype == EVT_APS:
        #     target_fh.write(s)

        aerdata_fh.seek(p)
        s = aerdata_fh.read(aeLen)  # read the first 8 byte
        p += aeLen

def extract_DVS_labels(nr_frames, frame_indexes):
    dvs_h5 = h5py.File(hdf5_name, 'r')

    labels = dvs_h5['labels'][-nr_frames:]
    labels = labels[frame_indexes]
    i=0
    label_dict = {"N": "1", "L": "2", "C": "3", "R": "4",}

    for label in labels:
        if label == 1:
            key = 'N'
        elif label == 2:
            key = 'L'
        elif label == 3:
            key = 'C'
        elif label == 4:
            key = 'R'
        label_dict[key] = str(label)

    jsonarray = json.dumps(label_dict)

    with open('./data/aedat/dvs_test_labels_' + str(args.recording) + '.json', 'w') as f:
        json.dump(label_dict, f)

def extract_k_random_frames(aedat_file, nr_frames, frame_size, k):
    aerdata_fh = open(aedat_file, 'rb')
    dvs_h5 = h5py.File(hdf5_name, 'r')

    if k == None or k > nr_frames:
        # use all frames
        k = nr_frames
        frame_indexes = np.arange(nr_frames)
    else:
        # use a subset of the frames
        # frame_indexes = np.random.randint(nr_frames, size=k)
        frame_indexes = np.arange(start=nr_frames, step=int(nr_frames/k), dtype=int)

    labels = dvs_h5['labels'][-nr_frames:]
    labels = labels[frame_indexes]

    # HEADER
    p, header_lines = parse_header(aerdata_fh)

    for nr, index in enumerate(frame_indexes):
        file_pointer = p + index*aeLen*frame_size
        aerdata_fh.seek(file_pointer)

        frame_data = aerdata_fh.read(frame_size*aeLen)
        
        if labels[nr] == 1:
            dir = 'N/'
        elif labels[nr] == 2:
            dir = 'L/'
        elif labels[nr] == 3:
            dir = 'C/'
        elif labels[nr] == 4:
            dir = 'R/'
        
        target_dir = os.path.dirname(aedat_file) + '/' + dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        target_path = target_dir + 'rec_' + str(args.recording) + '_sample_' + str(nr) + '.aedat'
        target_fh = open(target_path, 'wb')

        target_fh.writelines(header_lines)
        target_fh.write(frame_data)

    return frame_indexes






start_time = time.time()
# check_target(aedat_file, target_file)

# extract_DVS_events(aedat_file, full_target_file)

nr_events = write_test_aedat(full_target_file, test_target_file, TEST_FRACTION)
nr_frames = int(nr_events/EVENTS_PER_FRAME)

if NR_FRAME_DIV != None:
    k = int(NR_FRAME_DIV*nr_frames)
else:
    k = nr_frames

frame_indexes = extract_k_random_frames(test_target_file, nr_frames, 5000, k)

# extract_DVS_labels(nr_frames, frame_indexes)

print('total number of frames: ' + str(nr_frames))
print('extracted number of frames: ' + str(k))
print("--- %s seconds ---" % (time.time() - start_time))