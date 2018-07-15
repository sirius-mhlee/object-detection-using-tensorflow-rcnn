import sys
import cv2

import numpy as np
import tensorflow as tf

import AlexNet as an
import LinearSVM as svm
import BBoxRegression as bbr

import DataOperator as do

def print_batch_info(epoch_idx, batch_idx, loss_mean_value):
    print('Epoch : {0}, Batch : {1}, Loss Mean : {2}'.format(epoch_idx, batch_idx, loss_mean_value))

def print_epoch_info(epoch_idx, accuracy_mean_value):
    print('Epoch : {0}, Accuracy Mean : {1}'.format(epoch_idx, accuracy_mean_value))

def main():
    max_epoch = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    with tf.Session() as sess:
        train_data, train_mean = do.load_train_data(sys.argv[1])
        train_size = len(train_data)

        image = tf.placeholder(tf.float32, [None, 227, 227, 3])
        label = tf.placeholder(tf.float32, [None, 1000])
        bbox = tf.placeholder(tf.float32, [None, 4])

        # Build Image Classification Model
        alexnet_model = an.AlexNet(None, train_mean, True)
        with tf.name_scope('alexnet_content'):
            alexnet_model.build(image, label)

        # Build SVM Model
        linear_svm_model = svm.LinearSVM(None, True)
        with tf.name_scope('linear_svm_content'):
            linear_svm_model.build(alexnet_model.fc7, label)

        # Build Bounding Box Model
        bbox_model = bbr.BBoxRegression(None, True)
        with tf.name_scope('bbox_content'):
            bbox_model.build(alexnet_model.fc7, bbox)

        writer = tf.summary.FileWriter('./log/', sess.graph)

        # Training All

        #sess.run(tf.global_variables_initializer())

        #for epoch_idx in range(max_epoch):
        #    for batch_idx in range(train_size // batch_size):
        #        batch_image, batch_label = do.get_batch_data(sess, train_data, batch_size)

        #        alexnet_feed_dict = {image:batch_image, label:batch_label}
        #        _, loss_mean_value = sess.run([alexnet_model.optimizer, alexnet_model.loss_mean], feed_dict=alexnet_feed_dict)

        #        linear_svm_feed_dict = {label:batch_label}
        #        _, loss_mean_value = sess.run([linear_svm_model.optimizer, linear_svm_model.loss], feed_dict=linear_svm_feed_dict)
                
        #        print_batch_info(epoch_idx, batch_idx, loss_mean_value)

        #    batch_image, batch_label = do.get_batch_data(sess, train_data, batch_size)
        #    feed_dict = {image:batch_image, label:batch_label}

        #    accuracy_mean_value = sess.run(accuracy_mean, feed_dict=feed_dict)
        #    print_epoch_info(epoch_idx, accuracy_mean_value)

        ## Save All Model
        #do.save_model_npy(sess, alexnet_model.var_dict, sys.argv[4])
        #do.save_mean_npy(sess, alexnet_model.npy_mean, sys.argv[5])
        #do.save_model_npy(sess, linear_svm_model.var_dict, sys.argv[6])
        #do.save_model_npy(sess, bbox_model.var_dict, sys.argv[7])

if __name__ == '__main__':
    main()
