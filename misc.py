import numpy as np
import h5py
import keras

def three_sigma_frame_clipping(frame):
    # Compute standard deviation of event-sum distribution
    # after removing zeros
    sigma = np.std(frame[np.nonzero(frame)])

    # Clip number of events per pixel to three-sigma
    frame = np.clip(frame, 0, 3*sigma)


# scale frame pixel values to [0,1]
def frame_scaling(frame):
    min_pixel = np.min(frame)
    max_pixel = np.max(frame)

    frame = (frame - np.min(frame)) / float((np.max(frame) - np.min(frame)))


def batch_generator(frames, labels, batch_size, num_epochs, shuffle=False):
    # frames = np.array(data['images'][:])
    # labels = np.array(data['labels'][:])
    data_size = len(labels)

    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            frames = frames[shuffle_indices]
            labels = labels[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield frames[start_index:end_index], labels[start_index:end_index]

class data_iterator:
    def __init__(self, data, shuffle=False):
        self.df = data
        self.size = len(self.df['labels'])
        self.epochs = 0
        self.do_shuffle = shuffle
        self.cursor = 0
        if self.do_shuffle:
            self.shuffle()

    def shuffle(self):
        shuffle_indices = np.random.permutation(np.arange(self.size))
        self.df['images'][:] = self.df['images'][shuffle_indices]
        self.df['labels'][:] = self.df['labels'][shuffle_indices]
        self.cursor = 0

    def next_batch(self, batch_size):
        if self.cursor + batch_size >= self.size:
            frames = self.df['images'][self.cursor:]
            labels = self.df['labels'][self.cursor:]
            self.epochs += 1
            if self.do_shuffle:
                self.shuffle()
            return frames, labels
        else:
            frames = self.df['images'][self.cursor:self.cursor + batch_size - 1]
            labels = self.df['labels'][self.cursor:self.cursor + batch_size - 1]
            self.cursor += batch_size
            return frames, labels


def generate_batches_from_hdf5_file(hdf5_file, batch_size, dimensions, num_classes):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    filesize = len(hdf5_file['labels'])

    while 1:
        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries < (filesize - batch_size):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            xs = hdf5_file['images'][n_entries: n_entries + batch_size]
            xs = np.reshape(xs, dimensions).astype('float32')

            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...
            y_values = hdf5_file['labels'][n_entries:n_entries + batch_size]
            ys = keras.utils.to_categorical(y_values, num_classes)

            # we have read one more batch from this file
            n_entries += batch_size
            yield (xs, ys)
        # hdf5_file.close()


