#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import tensorflow as tf
import numpy as np
import logging
import sys
import time
import fcn8_vgg
import util
util.proc.set_proc_name('fcn');

from data import ICDARData
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf
import time
import util
util.proc.set_proc_name('fcn-test');
train_dir = '/home/dengdan/temp_nfs/tensorflow/fcn'
device = '/cpu:0'
data_dir = '/home/dengdan/temp_nfs/no-use/fsns'

with tf.Graph().as_default():
    with tf.device(device):
        with tf.Session() as sess:
            out_shape = [150, 150]
            images = tf.placeholder("float", name = 'images', shape = [None, None, 3])
            input_images = tf.expand_dims(images, 0)
            vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path='/home/dengdan/models/vgg/vgg16.npy')
            vgg_fcn.build(input_images, train = False)
            print('Finished building Network.')
            
            saver = tf.train.Saver();
            ckpt = tf.train.get_checkpoint_state(train_dir)
            saver.restore(sess, util.io.join_path(train_dir, ckpt.model_checkpoint_path))
            dump_dir = util.io.join_path(data_dir, util.io.get_filename(ckpt.model_checkpoint_path))
            print 'dumping into:', dump_dir
            print("Model restored from %s..."%(ckpt.model_checkpoint_path))

            image_names = util.io.ls(data_dir, '.jpg');            
            for name in image_names:
                print name
                image_path = util.io.join_path(data_dir, name)
                dump_path = util.io.join_path(dump_dir, name)
                image_data = util.img.imread(image_path, rgb = True)
                feed_dict = {images: image_data}
                score_map, logits = sess.run([vgg_fcn.pred_score, vgg_fcn.upscore8], feed_dict = feed_dict)
                S = score_map[0, ..., 1] > 0.5
                L = logits[0, ..., 1]
                util.plt.show_images(images = [image_data, S, L], titles = ["image", "%f"%(np.mean(S)), "Logits"], show = False, save = True, axis_off = True, path = dump_path)
