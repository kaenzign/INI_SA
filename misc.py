import numpy as np
import h5py

def three_sigma_frame_clipping_evtsum(frame):
    # Compute standard deviation of event-sum distribution
    # after removing zeros
    sigma = np.std(frame[np.nonzero(frame)])

    # Clip number of events per pixel to three-sigma
    frame = np.clip(frame, 0, 3*sigma)
    return frame



def aps_frame_scaling(frame):
    # scale frame pixel values to [0,1]
    min_pixel = np.min(frame)
    max_pixel = np.max(frame)

    frame = (frame - min_pixel) / float((max_pixel - min_pixel))
    return frame


def three_sigma_frame_clipping(frame):
    # Compute standard deviation of event-sum distribution
    sigma = np.std(frame)
    mean = np.mean(frame)

    # Clip number of events per pixel to three-sigma, but leave 0.5 entries unchanged as they correspond to zero event counts
    c_l = mean - 1.5*sigma
    c_r = mean + 1.5*sigma
    frame[np.nonzero(frame!=0.5)] = np.clip(frame[np.nonzero(frame!=0.5)], c_l, c_r)
    return frame


def dvs_frame_scaling(frame):
    # scale frame pixel values to [0,1], while remain 0.5 values unchanged
    min_pixel = np.min(frame)
    max_pixel = np.max(frame)

    frame[np.nonzero(frame != 0.5)] = (frame[np.nonzero(frame != 0.5)] - min_pixel) / float((max_pixel - min_pixel))
    return frame


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.int32)
    categorical[np.arange(n), y-1] = 1 # -1 because labels start at 1 not at 0
    return categorical

def generate_batches_from_hdf5_file(hdf5_file, batch_size, dimensions, num_classes, shuffle):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    """
    data_size = len(hdf5_file['labels'])
    indices = np.arange(data_size)

    while 1:
        if shuffle:
            indices = np.random.permutation(indices)

        # count how many entries we have read
        n_entries = 0
        # as long as we haven't read all entries from the file: keep reading
        while n_entries < (data_size - batch_size):
            # start the next batch at index 0
            # create numpy arrays of input data (features)
            if shuffle:
                # indices have to be in increasing order for hdf5 access (unlike numpy...)
                batch_indices = sorted(indices[n_entries: n_entries + batch_size])
            else:
                batch_indices = list(indices[n_entries: n_entries + batch_size])

            xs = hdf5_file['images'][batch_indices]
            xs = np.reshape(xs, dimensions).astype('float32')

            y_values = hdf5_file['labels'][batch_indices]
            #ys = keras.utils.to_categorical(y_values, num_classes)
            ys = to_categorical(y_values, num_classes)

            # we have read one more batch from this file
            n_entries += batch_size
            yield (xs, ys)

        # hdf5_file.close()

