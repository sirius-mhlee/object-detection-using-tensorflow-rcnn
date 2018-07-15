import cv2
import random as rand

import numpy as np
import tensorflow as tf

def load_npy(npy_path, mean_path):
    npy_file = np.load(npy_path, encoding='latin1').item()

    mean_file = open(mean_path, 'r')
    line = mean_file.readline()
    split_line = line.split(' ')
    npy_mean = [float(split_line[0]), float(split_line[1]), float(split_line[2])]
    mean_file.close()

    return npy_file, npy_mean

def save_npy(sess, var_dict, npy_mean, npy_path, mean_path):
    data_dict = {}

    for (name, idx), var in list(var_dict.items()):
        var_out = sess.run(var)
        if name not in data_dict:
            data_dict[name] = {}
        data_dict[name][idx] = var_out

    np.save(npy_path, data_dict)

    mean_file = open(mean_path, 'w')
    mean_file.write('{0} {1} {2}'.format(npy_mean[0], npy_mean[1], npy_mean[2]))
    mean_file.close()

def load_image(img_path):
    img = cv2.imread(img_path)
    reshape_img = cv2.resize(img, dsize=(227, 227), interpolation=cv2.INTER_CUBIC)
    np_img = np.asarray(reshape_img, dtype=float)
    expand_np_img = np.expand_dims(np_img, axis=0)
    return expand_np_img

def load_train_data(train_data_path):
    train_data = []
    train_mean = [0, 0, 0]

    train_file = open(train_data_path, 'r')
    all_line = train_file.readlines()
    for line in all_line:
        split_line = line.split(' ')
        train_data.append((split_line[0], int(split_line[1])))

        image = load_image(split_line[0])
        train_mean += np.mean(image, axis=(0, 1, 2))
    train_file.close()

    train_mean /= len(train_data)

    return train_data, train_mean

def get_batch_data(sess, train_data, batch_size):
    rand.shuffle(train_data)
    
    image = []
    label = []

    batch_data = train_data[:batch_size]
    for data in batch_data:
        image.append(load_image(data[0]))
        label.append(data[1])

    batch_image = np.concatenate(image)
    batch_label_op = tf.one_hot(label, on_value=1, off_value=0, depth=1000)
    batch_label = sess.run(batch_label_op)

    return batch_image, batch_label
    