"""
Some utilities to load preprocessed images from disk and plot them.
Modified version of script by: https://www.kaggle.com/gauss256
"""
import glob
import os
import re
import gc

import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Processing parameters
SIZE = 224  # for ImageNet models compatibility
TEST_DIR = './datasets/asirra/preprocessed/test/'
TRAIN_DIR = './datasets/asirra/preprocessed/train/'
BASE_DIR = './datasets/asirra/preprocessed/'


def natural_key(string_):
    """
    Define sort key that is integer-aware
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def prep_images(paths):
    """
    Preprocess images
    """
    out = list()
    for count, path in enumerate(paths):
        img = Image.open(path)
        img_arr = np.asarray(img)
        img.close()
        pixel_height = img_arr.shape[0]
        pixel_width = img_arr.shape[0]
        dim_count = img_arr.shape[2]
        img_arr = img_arr.reshape(pixel_height*pixel_width*dim_count) # reshape to 1d vector
        out.append(img_arr)
    return out

def load_data(shuffle_and_split=True, memmap=False):
    if shuffle_and_split:
        data, target = load_data(shuffle_and_split=False, memmap=False)
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
        if memmap:
            return (as_memmap(X_train), y_train), (as_memmap(X_test), y_test)
        else:
            return (X_train, y_train), (X_test, y_test)

    # Get the paths to all the image files
    train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')), key=natural_key)
    train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')), key=natural_key)

    obs_count = len(train_cats)+len(train_dogs)
    feat_count = 224*224*3

    # reserve some memory - 1 byte per pixel
    # (1 byte * 244 * 244 * 3) * 25 000 = 4.4652 gigabytes
    data = np.empty((obs_count,feat_count), dtype=np.uint8)
    data[:len(train_cats),:] = prep_images(train_cats)
    data[len(train_cats):,:] = prep_images(train_dogs)

    cats_target, dogs_target = np.zeros((len(train_cats),),dtype=np.uint8), np.ones((len(train_dogs),),dtype=np.uint8)
    target = np.concatenate((cats_target,dogs_target))

    if memmap:
        return (as_memmap(data), target)
    else:
        return (data, target)

def load_data_imbalanced(ratio=0.1, shuffle_and_split=True, memmap=True):
    if shuffle_and_split:
        data, target = load_data_imbalanced(ratio=ratio, shuffle_and_split=False, memmap=False)
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=ratio)
        if memmap:
            return (as_memmap(X_train), y_train), (as_memmap(X_test), y_test)
        else:
            return (X_train, y_train), (X_test, y_test)
    
    data, target = load_data(shuffle_and_split=False, memmap=False)
    minority = np.where(target==1)[0]
    minority_count = int(np.floor( ratio * len(minority) ))
    minority = np.random.choice(minority, size=minority_count)
    minority = data[minority]
    majority = data[target==0]
    # overwrite data and target with their imbalanced versions
    data = np.concatenate((majority, minority))

    target = np.concatenate( (np.zeros(majority.shape[0], dtype=np.uint8), np.ones(minority.shape[0], dtype=np.uint8)) )
    if memmap:
        return (as_memmap(data), target)
    else:
        return (data, target)  

def as_memmap(data):
    shape = data.shape
    # write from array into memmap
    filename = 'asirra.memmap'
    i = 2
    try:
        data_memmap = np.memmap(filename=filename,dtype=data.dtype, mode='w+', shape=shape)
    except:
        while os.path.isfile(filename):
            filename = 'asirra.{}.memmap'.format(str(i))
            i += 1
        data_memmap = np.memmap(filename=filename,dtype=data.dtype, mode='w+', shape=shape)
    
    #data_memmap[:] = data[:]
    np.copyto(data_memmap, data)
    # flush data and memmap
    del data_memmap, data
    # open memmap in read mode
    data_memmap = np.memmap(filename=filename,dtype=np.uint8, mode='r', shape=shape)
    return data_memmap
    
def plot(data, labels=np.zeros((0,)), count=-1):
    data = np.asarray(data)
    labels = np.asarray(labels)
    if data.size < 2:
        print('Can\'t plot: empty input data.')
        return
    if labels.size is 0:
        labels = np.ones(data.shape[0]) * -1    
    if count > 0:
        if data.shape[0] < count: count = data.shape[0]
        data, labels = data[0:count], labels[0:count]
    # split into chunks of 5
    for (data_portion, labels_portion) in [(data[i:i+5 ], labels[i:i+5]) for i in range(0, data.shape[0], 5)]:
        plt.figure(figsize=(10,5))
        for i, (obs,label) in enumerate(zip(data_portion,labels_portion)):
            # Plot
            ax = plt.subplot(2, data_portion.shape[0], i + 1)
            # if label >= 0:
            plt.title(label)
            plt.grid(False)
            plt.axis('off')
            plt.imshow(obs.reshape(224,224,3))
        plt.show()