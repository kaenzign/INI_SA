import h5py
from os import listdir
from os.path import isfile, join

processed_path = './data/processed/'
filenames = [f for f in listdir(processed_path) if isfile(join(processed_path, f))]


train_output_file = h5py.File('./data/train.hdf5', 'w')


#keep track of the total number of of frames
nr_frames = 0

for n, filename in enumerate(filenames):
    file_path = processed_path + filename
    hdf5_f = h5py.File(file_path,'r')
    nr_frames += hdf5_f['images'].shape[0]

    if n == 0:
        #first file; create the dummy dataset with no max shape
        FRAME_DIM = (hdf5_f['images'].shape[1], hdf5_f['images'].shape[2])
        image_dataset = train_output_file.create_dataset("images", (nr_frames, FRAME_DIM[0], FRAME_DIM[1]), maxshape=(None, FRAME_DIM[0], FRAME_DIM[1]))
        label_dataset = train_output_file.create_dataset("labels", (nr_frames, ), maxshape=(None, ))
        #fill the first section of the dataset
        image_dataset[:] = hdf5_f['images'][:]
        label_dataset[:] = hdf5_f['labels'][:]
        where_to_start_appending = nr_frames

    else:
        #resize the dataset to accomodate the new data
        image_dataset.resize(nr_frames, axis=0)
        label_dataset.resize(nr_frames, axis=0)

        image_dataset[where_to_start_appending:nr_frames] = hdf5_f['images'][:]
        label_dataset[where_to_start_appending:nr_frames] = hdf5_f['labels'][:]
        where_to_start_appending = nr_frames

output_file.close()