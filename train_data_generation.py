import h5py
from os import listdir
from os.path import isfile, join

EULER = True

if EULER:
    processed_path = '../../../scratch/kaenzign/processed/'
else:
    processed_path = './data/processed/'

processed_path += 'aps_36/'

filenames = [f for f in listdir(processed_path) if isfile(join(processed_path, f))]


train_output_file = h5py.File(processed_path + 'train.hdf5', 'w')
test_output_file = h5py.File(processed_path + 'test.hdf5', 'w')


#keep track of the total number of of frames
nr_frames = 0
tot_nr_train_frames = 0
tot_nr_test_frames = 0

# filenames = [filenames[0], filenames[1]] # TODO: remove.. just for local debug

for n, filename in enumerate(filenames):
    print('PROCESSING FILE ' + str(n) + ' ' + filename)
    file_path = processed_path + filename
    hdf5_f = h5py.File(file_path,'r')

    current_nr_frames = hdf5_f['images'].shape[0]
    current_nr_test_frames = int(0.2*current_nr_frames)
    current_nr_train_frames = current_nr_frames - current_nr_test_frames
    tot_nr_test_frames += current_nr_test_frames
    tot_nr_train_frames += current_nr_train_frames

    if n == 0:
        #first file; create the dummy dataset with no max shape
        FRAME_DIM = (hdf5_f['images'].shape[1], hdf5_f['images'].shape[2])
        train_image_dataset = train_output_file.create_dataset("images", (tot_nr_train_frames, FRAME_DIM[0], FRAME_DIM[1]), maxshape=(None, FRAME_DIM[0], FRAME_DIM[1]))
        train_label_dataset = train_output_file.create_dataset("labels", (tot_nr_train_frames, ), maxshape=(None, ))
        test_image_dataset = test_output_file.create_dataset("images", (tot_nr_test_frames, FRAME_DIM[0], FRAME_DIM[1]), maxshape=(None, FRAME_DIM[0], FRAME_DIM[1]))
        test_label_dataset = test_output_file.create_dataset("labels", (tot_nr_test_frames, ), maxshape=(None, ))
        #fill the first section of the dataset
        train_image_dataset[:] = hdf5_f['images'][:current_nr_train_frames]
        train_label_dataset[:] = hdf5_f['labels'][:current_nr_train_frames]
        test_image_dataset[:] = hdf5_f['images'][-current_nr_test_frames:]
        test_label_dataset[:] = hdf5_f['labels'][-current_nr_test_frames:]
        where_to_append_train = tot_nr_train_frames
        where_to_append_test = tot_nr_test_frames

    else:
        #resize the dataset to accomodate the new data
        train_image_dataset.resize(tot_nr_train_frames, axis=0)
        train_label_dataset.resize(tot_nr_train_frames, axis=0)
        test_image_dataset.resize(tot_nr_test_frames, axis=0)
        test_label_dataset.resize(tot_nr_test_frames, axis=0)

        train_image_dataset[where_to_append_train:tot_nr_train_frames] = hdf5_f['images'][:current_nr_train_frames]
        train_label_dataset[where_to_append_train:tot_nr_train_frames] = hdf5_f['labels'][:current_nr_train_frames]
        test_image_dataset[where_to_append_test:tot_nr_test_frames] = hdf5_f['images'][-current_nr_test_frames:]
        test_label_dataset[where_to_append_test:tot_nr_test_frames] = hdf5_f['labels'][-current_nr_test_frames:]
        where_to_append_train = tot_nr_train_frames
        where_to_append_test = tot_nr_test_frames

train_output_file.close()
hdf5_f.close()