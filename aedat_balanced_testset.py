import os
import numpy as np
from shutil import copyfile
import random

NR_SAMPLES_PER_CLASS = 100

def get_sample_filepaths_and_labels(dataset_path):
    # Count the number of samples and classes
    classes = [subdir for subdir in sorted(os.listdir(dataset_path))
               if os.path.isdir(os.path.join(dataset_path, subdir))]

    label_dict = label_dict = {"N": "0", "L": "1", "C": "2", "R": "3",}
    num_classes = len(label_dict)
    assert num_classes == len(classes), \
        "The number of classes provided by label_dict {} does not match " \
        "the number of subdirectories found in dataset_path {}.".format(
            label_dict, dataset_path)

    class_filenames = [[],[],[],[]]
    labels = []
    num_samples = 0
    for subdir in classes:
        for fname in sorted(os.listdir(os.path.join(dataset_path, subdir))):
            is_valid = False
            for extension in {'aedat'}:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                labels.append(label_dict[subdir])
                class_filenames[int(label_dict[subdir])].append(os.path.join(dataset_path, subdir, fname))
                num_samples += 1
    labels = np.array(labels, 'int32')
    print("Found {} samples belonging to {} classes.".format(
        num_samples, num_classes))
    return class_filenames, labels




filepaths, labels = get_sample_filepaths_and_labels('./data/aedat/cut_0.05/')


nr_class_samples = [0,0,0,0]
for label, path_list in enumerate(filepaths):
    nr_class_samples[label] = len(path_list)

min_class_samples = min(nr_class_samples)

if min_class_samples < NR_SAMPLES_PER_CLASS:
    NR_SAMPLES_PER_CLASS = min_class_samples


extracted_class_paths = [[],[],[],[]]
for label, path_list in enumerate(filepaths):
    path_indexes = random.sample(range(len(path_list)), NR_SAMPLES_PER_CLASS)
    extracted_class_paths[label] = np.array(path_list)[path_indexes]

DST_PATH = './AEDAT_TESTSET/'
DST_PATH = os.path.dirname(DST_PATH)

try:
    os.stat(DST_PATH)
except:
    os.mkdir(DST_PATH)

DST_CLASS_PATHS = [DST_PATH + '/N/', DST_PATH + '/L/', DST_PATH + '/C/', DST_PATH + '/R/']

os.mkdir(DST_CLASS_PATHS[0])
os.mkdir(DST_CLASS_PATHS[1])
os.mkdir(DST_CLASS_PATHS[2])
os.mkdir(DST_CLASS_PATHS[3])

k = 0
for i, class_paths in enumerate(extracted_class_paths):
    for path in class_paths:
        copyfile(path, DST_CLASS_PATHS[i] + os.path.basename(path))
        k = k +1
        print(i, k, DST_CLASS_PATHS[i] + os.path.basename(path))
i = 0

