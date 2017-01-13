# pip3 install matplotlib
# pip3 install numpy
# pip install --upgrade "ipython[all]"
# pip3 install scipy
# pip3 install sklearn
# pip3 install Pillow
# pip3 install jupyter

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import pickle
import string

from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve

import random
import collections
from operator import itemgetter
from hashlib import md5

# Config the matplotlib backend as plotting inline in IPython
# %matplotlib inline

# Set working directory to tmp to story all data & pickle files
def set_working_dir():
    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    os.chdir(tmp_dir)
    print("Change working directory to", os.getcwd())

set_working_dir()

do_plotting = True
force = False

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('\nAttempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)

num_classes = 10
np.random.seed(133)


def maybe_extract(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print('Extracted dir:', root, data_folders)
    return data_folders


print('Locate or extract dataset:')
test_folders = maybe_extract(test_filename)
train_folders = maybe_extract(train_filename)


def display_samples(folders):
    """If run in ipython, display a sample image for each folder.
    If it is not run in ipython, display a string only.
    """
    if not do_plotting:
        return
    for folder in folders:
        print(folder)
        image_files = os.listdir(folder)
        image = random.choice(image_files)
        image_file = os.path.join(folder, image)
        i = Image(filename=image_file)
        display(i)


print('Display random image for each class:')
display_samples(test_folders)
# display_samples(train_folders)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print('Loading data for the class folder:', folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

print('Pickle train and test images for each class if needed:')
test_datasets = maybe_pickle(test_folders, 1800)
train_datasets = maybe_pickle(train_folders, 45000)

print('Testing datasets: ', test_datasets)
print('Train datasets: ', train_datasets)


def display_letters(letters, labels, figure_name):
    if not do_plotting:
        return
    fig = plt.figure()
    fig.suptitle(figure_name + ' (close this window to continue)', fontsize=20)
    total = len(letters)
    for index, image in enumerate(letters):
        a = fig.add_subplot(1, total, index + 1)
        plt.imshow(image)
        a.set_title(labels[index])
        a.axis('off')
    plt.show()


def display_each_letter(datasets):
    letters = []
    labels = []
    for index, pickle_name in enumerate(datasets):
        print('Loading %s.' % pickle_name)
        try:
            with open(pickle_name, 'rb') as f:
                images = pickle.load(f)
                pos = random.randrange(0, len(images))
                letters.append(images[pos])
                label = pickle_name.split('/')[1][0]
                labels.append(label)
        except Exception as e:
            print('Unable to load data from', pickle_name, ':', e)
    figure_name = datasets[0].split('/')[0]
    display_letters(letters, labels, figure_name)


print('Load and display 1 letter per class:')
display_each_letter(test_datasets)
display_each_letter(train_datasets)


def verify_balance(datasets):
    for index, pickle_name in enumerate(datasets):
        try:
            with open(pickle_name, 'rb') as f:
                images = pickle.load(f)
                print("Length %s = %s" % (pickle_name, len(images)))
        except Exception as e:
            print('Unable to load data from', pickle_name, ':', e)

print('Verify if each class contains similar amount of images:')
verify_balance(test_datasets)
verify_balance(train_datasets)


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Dataset shape before sanitize:')
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


def letter(i):
    return string.ascii_lowercase[:26][i]


def sample_data(dataset, labels, size, figure_name):
    rand_index = random.sample(range(len(dataset)), size)
    sample_data = [dataset[i] for i in rand_index]
    sample_label = [letter(labels[i]) for i in rand_index]
    display_letters(sample_data, sample_label, figure_name)


sample_data(test_dataset, test_labels, 10, 'Testing data')
sample_data(train_dataset, train_labels, 10, 'Training data')


def label_count(labels):
    c = collections.Counter(labels)
    c = dict((letter(key), value) for (key, value) in c.items())
    return sorted(c.items(), key=itemgetter(0))

print('Count for each class:')
print('Test dataset:', label_count(test_labels))
print('Train dataset:', label_count(train_labels))

def sanitize(dataset1, dataset2, labels1):
    hash1 = np.array([md5(d).hexdigest() for d in dataset1])
    hash2 = np.array([md5(d).hexdigest() for d in dataset2]) if dataset2 is not None else None
    seen = []
    overlap = []
    for i, value in enumerate(hash1):
        is_overlap = len(np.where(hash2 == value)[0]) if dataset2 is not None else False
        if is_overlap or value in seen:
            overlap.append(i)
        seen.append(value)
    return np.delete(dataset1, overlap, 0), np.delete(labels1, overlap, None)

pickle_file = 'notMNIST.pickle'

if not os.path.exists(pickle_file):
    print('Computing MD5 ...')
    set_test_dataset = set([md5(d).hexdigest() for d in test_dataset])
    set_valid_dataset = set([md5(d).hexdigest() for d in valid_dataset])
    set_train_dataset = set([md5(d).hexdigest() for d in train_dataset])

    unique_test_dataset = set_test_dataset - set_valid_dataset - set_train_dataset
    unique_valid_dataset = set_valid_dataset - set_test_dataset - set_train_dataset
    unique_train_dataset = set_train_dataset - set_test_dataset - set_valid_dataset

    print('Compute overlap:')
    print('Train dataset: ' + str(len(train_dataset)) + ' set: ' + str(len(set_train_dataset)) + ' unique: ' + str(
        len(unique_train_dataset)))
    print('Valid dataset: ' + str(len(valid_dataset)) + ' set: ' + str(len(set_valid_dataset)) + ' unique: ' + str(
        len(unique_valid_dataset)))
    print('Test dataset: ' + str(len(test_dataset)) + ' set: ' + str(len(set_test_dataset)) + ' unique: ' + str(
        len(unique_test_dataset)))

    print('Sanitize training data ... may take a while.')
    train_dataset, train_labels = sanitize(train_dataset, None, train_labels)
    print('Sanitize testing data ... may take a while.')
    test_dataset, test_labels = sanitize(test_dataset, train_dataset, test_labels)
    test_dataset, test_labels = sanitize(test_dataset, valid_dataset, test_labels)
    print('Sanitize validation data ... may take a while.')
    valid_dataset, valid_labels = sanitize(valid_dataset, train_dataset, valid_labels)
    valid_dataset, valid_labels = sanitize(valid_dataset, test_dataset, valid_labels)

    print('After sanitize:')
    print('Train dataset: ' + str(len(train_dataset)) + ' labels: ' + str(len(train_labels)))
    print('Valid dataset: ' + str(len(valid_dataset)) + ' labels: ' + str(len(valid_labels)))
    print('Test dataset: ' + str(len(test_dataset)) + ' labels: ' + str(len(test_labels)))


def maybe_final_pickle(all_dataset):
    if not force and os.path.exists(pickle_file):
        print('%s already present - Skipping.' % pickle_file)
    else:
        try:
            f = open(pickle_file, 'wb')
            pickle.dump(all_dataset, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)


save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
}

maybe_final_pickle(save)


def load_final_pickle():
    datasets = {}
    try:
        with open(pickle_file, 'rb') as f:
            datasets = pickle.load(f)
    except Exception as ex:
        print(ex)
    return datasets


save = load_final_pickle()

print('Checking dataset shape before training:')
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

print("Train & testing: 50, 100, 1000, 5000 model (5000 will take a while)")

train_dataset = save['train_dataset']
train_labels = save['train_labels']

valid_dataset = save['valid_dataset']
valid_labels = save['valid_labels']

test_dataset = save['test_dataset'].reshape(save['test_dataset'].shape[0], image_size * image_size)
test_labels = save['test_labels']

def create_model(dataset, labels, size):
    X_train = dataset[:size].reshape(size, image_size * image_size)
    y_train = labels[:size]
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


m50 = create_model(train_dataset, train_labels, 50)
print('Score for model 50:', m50.score(test_dataset, test_labels))

m100 = create_model(train_dataset, train_labels, 100)
print('Score for model 100:', m100.score(test_dataset, test_labels))

m1000 = create_model(train_dataset, train_labels, 1000)
print('Score for model 1000: ', m1000.score(test_dataset, test_labels))

m5000 = create_model(train_dataset, train_labels, 5000)
print('Score for model 5000: ', m5000.score(test_dataset, test_labels))

pred_labels = m5000.predict(test_dataset)
sample_data(save['test_dataset'], pred_labels, 10, 'Prediction on test data')
