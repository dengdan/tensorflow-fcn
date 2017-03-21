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
from preprocessing.preprocessing_factory  import get_preprocessing
import numpy as np
import tensorflow as tf
import time
import util
util.proc.set_proc_name('fcn');
fn = get_preprocessing(True);

learning_rate = 1e-8
momentum = 0.9
weight_decay = 5e-4
max_steps = 10000000
train_dir = '/home/dengdan/temp_nfs/tensorflow/fcn'
device = '/cpu:0'


with tf.Graph().as_default():
    with tf.device(device):
        with tf.Session() as sess:
            out_shape = [150, 150]
            images = tf.placeholder("float", name = 'images', shape = [None, None, 3])
            bboxes = tf.placeholder("float", name = 'bboxes', shape = [1, None, 4])
            labels = tf.placeholder('int32', name = 'labels', shape = [None, 1])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            
            sampled_image, sampled_mask, sampled_bboxes = fn(images, labels, bboxes, out_shape);
            sampled_images = tf.expand_dims(sampled_image, 0)
            sampled_masks = tf.expand_dims(sampled_mask, 0)        
            vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path='/home/dengdan/models/vgg/vgg16.npy', weight_decay = weight_decay)
            vgg_fcn.build(sampled_images, labels = sampled_masks, debug=True, train = True)
            opt_vars = [];
                    
            static_layers = ['conv1', 'conv2', 'conv3', 'conv4']
            for v in tf.trainable_variables():
                if util.str.starts_with(v.name, static_layers):
                    continue
                opt_vars.append(v);
                print "%s is to be trained."%(v.name)
            
            opt = tf.train.MomentumOptimizer(learning_rate, momentum = momentum)
            train_op = opt.minimize(vgg_fcn.loss, var_list=opt_vars, global_step= global_step, name='train')
            print('Finished building Network.')
            
            saver = tf.train.Saver();
            
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, util.io.join_path(train_dir, ckpt.model_checkpoint_path))
                print("Model restored...")
            else:
                init_op = tf.global_variables_initializer();
                sess.run(init_op)
                print("Model initialized...")
            step = 0
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            merged = tf.summary.merge_all()
            data_provider = ICDARData();
            while step < max_steps:
                start = time.time();
                image_data, bbox_data, label_data = data_provider.get_data();
                feed_dict = {images: image_data, labels: label_data, bboxes: bbox_data}
                I, M, sb, summary, _, loss, step, pred, score = sess.run([sampled_image, sampled_mask, sampled_bboxes, merged, train_op, vgg_fcn.loss, global_step, vgg_fcn.pred_up, vgg_fcn.pred_score], feed_dict = feed_dict);
                #util.cit(I, name = 'image')
                #util.sit(M, name = 'label')
                #util.sit(pred[0, ...], name = 'pred')
                #util.sit(score[0, ..., 1], name = 'score');
                #util.plt.show_images(titles = ['image', 'label', 'pred', 'score_%f'%(np.mean(score[0, ..., 1]))], images = [np.asarray(I, dtype = np.uint8), M, pred[0, ...], score[0, ..., 1]], show = False, save = True, path = '~/temp_nfs/no-use/%d.jpg'%(step));
                summary_writer.add_summary(summary, step)
                end = time.time();
                print "Step %d, loss = %f, time used:%s seconds"%(step, loss, end - start)          
                if (step + 1)%5000 == 0:
                    saver.save(sess, util.io.join_path(train_dir, 'vgg16-fcn8s-iter-%d.ckpt'%(step + 1)));
                    
