import sys
import cv2

import numpy as np
import tensorflow as tf

import Configuration as cfg

import AlexNet as an
import LinearSVM as svm
import BBoxRegression as bbr

import DataOperator as do

def print_batch_info(epoch_idx, batch_idx, loss_mean_value):
    print('Epoch : {0}, Batch : {1}, Loss Mean : {2}'.format(epoch_idx, batch_idx, loss_mean_value))

def print_epoch_info(epoch_idx, accuracy_mean_value):
    print('Epoch : {0}, Accuracy Mean : {1}'.format(epoch_idx, accuracy_mean_value))

def main():
    max_epoch = int(sys.argv[3])
    batch_size = int(sys.argv[4])

    with tf.Session() as sess:
        alexnet_train_data, alexnet_train_mean = do.load_alexnet_train_data(sys.argv[1])
        alexnet_train_size = len(alexnet_train_data)

        alexnet_finetune_data = do.load_alexnet_finetune_data(sys.argv[2])
        alexnet_finetune_size = len(alexnet_finetune_data)

        svm_train_data = do.load_svm_train_data(sys.argv[2])
        svm_train_size = len(svm_train_data)

        bbox_train_data = do.load_bbox_train_data(sys.argv[2])
        bbox_train_size = len(bbox_train_data)

        image = tf.placeholder(tf.float32, [None, cfg.image_size_width, cfg.image_size_height, 3])
        label = tf.placeholder(tf.float32, [None, cfg.object_class_num])
        finetune_label = tf.placeholder(tf.float32, [None, cfg.object_class_num + 1])
        bbox = tf.placeholder(tf.float32, [None, 4])
        feature = tf.placeholder(tf.float32, [None, 4096])

        alexnet_model = an.AlexNet(None, alexnet_train_mean, True)
        with tf.name_scope('alexnet_content'):
            alexnet_model.build(image, label)
        with tf.name_scope('alexnet_finetune_content'):
            alexnet_model.build_finetune(finetune_label)

        svm_model = svm.LinearSVM(None, True)
        with tf.name_scope('svm_content'):
            svm_model.build(feature, finetune_label)

        bbox_model = bbr.BBoxRegression(None, True)
        with tf.name_scope('bbox_content'):
            bbox_model.build(feature, bbox)

        writer = tf.summary.FileWriter('./log/', sess.graph)

        sess.run(tf.global_variables_initializer())

        print('Training AlexNet')
        for epoch_idx in range(max_epoch):
            for batch_idx in range(alexnet_train_size // batch_size):
                batch_image, batch_label = do.get_alexnet_train_batch_data(sess, alexnet_train_data, batch_size)
                feed_dict = {image:batch_image, label:batch_label}

                _, loss_mean_value = sess.run([alexnet_model.optimizer, alexnet_model.loss_mean], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_mean_value)

            batch_image, batch_label = do.get_alexnet_train_batch_data(sess, alexnet_train_data, batch_size)
            feed_dict = {image:batch_image, label:batch_label}

            accuracy_mean_value = sess.run(alexnet_model.accuracy_mean, feed_dict=feed_dict)
            print_epoch_info(epoch_idx, accuracy_mean_value)

        print('Finetuning AlexNet')
        for epoch_idx in range(max_epoch):
            for batch_idx in range(alexnet_finetune_size // batch_size):
                batch_image, batch_label = do.get_alexnet_finetune_batch_data(sess, alexnet_finetune_data, batch_size)
                feed_dict = {image:batch_image, finetune_label:batch_label}

                _, loss_mean_value = sess.run([alexnet_model.finetune_optimizer, alexnet_model.finetune_loss_mean], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_mean_value)

            batch_image, batch_label = do.get_alexnet_finetune_batch_data(sess, alexnet_finetune_data, batch_size)
            feed_dict = {image:batch_image, finetune_label:batch_label}

            accuracy_mean_value = sess.run(alexnet_model.finetune_accuracy_mean, feed_dict=feed_dict)
            print_epoch_info(epoch_idx, accuracy_mean_value)

        print('Training Linear SVM')
        for epoch_idx in range(max_epoch):
            for batch_idx in range(svm_train_size // batch_size):
                batch_image, batch_label = do.get_svm_train_batch_data(sess, svm_train_data, batch_size)
                batch_feature_feed_dict = {image:batch_image}
                batch_feature = sess.run(alexnet_model.fc7, feed_dict=batch_feature_feed_dict)
                feed_dict = {feature:batch_feature, finetune_label:batch_label}

                _, loss_value = sess.run([svm_model.optimizer, svm_model.loss], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_value)

            batch_image, batch_label = do.get_svm_train_batch_data(sess, svm_train_data, batch_size)
            batch_feature_feed_dict = {image:batch_image}
            batch_feature = sess.run(alexnet_model.fc7, feed_dict=batch_feature_feed_dict)
            feed_dict = {feature:batch_feature, finetune_label:batch_label}

            accuracy_mean_value = sess.run(svm_model.accuracy_mean, feed_dict=feed_dict)
            print_epoch_info(epoch_idx, accuracy_mean_value)

        print('Training BBox Regression')
        for epoch_idx in range(max_epoch):
            for batch_idx in range(bbox_train_size // batch_size):
                batch_image, batch_bbox = do.get_bbox_train_batch_data(sess, bbox_train_data, batch_size)
                batch_feature_feed_dict = {image:batch_image}
                batch_feature = sess.run(alexnet_model.fc7, feed_dict=batch_feature_feed_dict)
                feed_dict = {feature:batch_feature, bbox:batch_bbox}

                _, loss_mean_value = sess.run([bbox_model.optimizer, bbox_model.loss_mean], feed_dict=feed_dict)
                print_batch_info(epoch_idx, batch_idx, loss_mean_value)

        do.save_model(sess, alexnet_model.var_dict, sys.argv[5])
        do.save_mean(alexnet_model.mean, sys.argv[6])
        do.save_model(sess, svm_model.var_dict, sys.argv[7])
        do.save_model(sess, bbox_model.var_dict, sys.argv[8])

if __name__ == '__main__':
    main()
