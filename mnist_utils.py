import matplotlib.pyplot as plt
import pickle, gzip
import numpy as np

def plot_mnist(data, labels=np.zeros((0,)), count=-1):#, reconstruction=[]):
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
    # split into 20 chunks
    for (data_portion, labels_portion) in [(data[i:i+20 ], labels[i:i+20]) for i in range(0, data.shape[0], 20)]:
        plt.figure(figsize=(20,2))
        for i, (obs,label) in enumerate(zip(data_portion,labels_portion)):#,reconstruction):
            pixels = np.array((obs * 255 - 255) * -1, dtype='uint8')
            pixels = pixels.reshape((28, 28))
            # Plot
            ax = plt.subplot(2, data_portion.shape[0], i + 1)
            # if label >= 0:
            plt.title(label)
            plt.grid(False)
            plt.axis('off')
            plt.imshow(pixels, cmap='gray')
        plt.show()

def load_data():
    f = gzip.open('./datasets/mnist.pkl.gz', 'rb')
    (train_set,train_set_target), (validation_set,validation_set_target) = [(data, target) for (data,target) in pickle.load(f, encoding='latin1')]
    f.close()
    # vectorize and normalize
    train_set, validation_set = ( np.reshape(data, (data.shape[0],data.shape[1]*data.shape[2])) / 255 for data in (train_set, validation_set))
    return (train_set,train_set_target), (validation_set,validation_set_target)

def load_binary_imbalanced(classes=(1,7), ratio=0.1):
    """ Return MNIST data imbalanced. First class will be majority class, second class will be minority class"""
    (train_set,train_set_target), (validation_set,validation_set_target) = load_data()
    
    # binarize
    mask_train_set_imb = np.logical_or(train_set_target  == classes[0],train_set_target  == classes[1])
    mask_validation_set_imb = np.logical_or(validation_set_target  == classes[0],validation_set_target  == classes[1])
    (train_set_imb,train_set_imb_target), (validation_set_imb, validation_set_imb_target) = (train_set[mask_train_set_imb], train_set_target[mask_train_set_imb]), (validation_set[mask_validation_set_imb], validation_set_target[mask_validation_set_imb])

    # imbalance
    for i, (data_set_imb, data_set_imb_target) in enumerate([(train_set_imb,train_set_imb_target), (validation_set_imb, validation_set_imb_target)]):
        data_minority = data_set_imb[data_set_imb_target == classes[1]]
        data_minority_target = data_set_imb_target[data_set_imb_target == classes[1]]
        data_majority = data_set_imb[data_set_imb_target == classes[0]]
        data_majority_target = data_set_imb_target[data_set_imb_target == classes[0]]
        original_size = data_minority_target.shape[0]
        majority_size = data_majority_target.shape[0]
        target_size = int(np.floor(majority_size * ratio))
        indices = np.random.choice(original_size, size=target_size)
        data_minority = data_minority[indices]
        data_minority_target = data_minority_target[indices]

        # merge
        if i == 0:
            train_set = np.concatenate([data_minority, data_majority])
            train_set_target = np.concatenate([data_minority_target, data_majority_target])
        else:
            validation_set = np.concatenate([data_minority, data_majority])
            validation_set_target = np.concatenate([data_minority_target, data_majority_target])

    #shuffle
    train_set, train_set_target = np.hsplit(
        np.random.permutation(
            np.hstack((train_set, train_set_target.reshape((train_set_target.shape[0], 1))))
        ), [-1]
    )
    validation_set, validation_set_target = np.hsplit(
        np.random.permutation(
            np.hstack((validation_set, validation_set_target.reshape((validation_set_target.shape[0],1))))
        ), [-1]
    )
    train_set_target, validation_set_target = np.asarray(train_set_target, dtype='int').reshape((train_set_target.shape[0],)), np.asarray(validation_set_target, dtype='int').reshape((validation_set_target.shape[0],))
    return (train_set,train_set_target), (validation_set,validation_set_target)