#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: t-sne.py
# Date: 2019 2019年06月10日 14:33:51
# Encoding: utf-8
# Author: Jingyi Xiao

__author__="Jingyi"

import os, sys, time
import argparse
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
FLAGS = None

def embeddingShow():
    data = np.load(FLAGS.input)
    idx = np.arange(len(data))
    if FLAGS.random:
        np.random.shuffle(idx)
    idx = idx[:FLAGS.nmax]

    tsv_data = None
    if FLAGS.tsv and os.path.exists(FLAGS.tsv):
        with open(FLAGS.tsv, 'r') as f:
            d = f.read()
        tsv_data = d.split('\n')
        tsv_data = list(filter(lambda x: x, tsv_data))
        assert(len(data) == len(tsv_data)-1), (data.shape, len(tsv_data))

        tsv_data = np.array(tsv_data)
        tsv_choose = tsv_data[idx+1]
        tsv_data = np.concatenate((np.array([tsv_data[0]]), tsv_choose), axis=0)

    data = data[idx]
    sess = tf.InteractiveSession()

    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.stack(data, axis=0),
                trainable=False, name='embedding')

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.logdir + '/projector', sess.graph)

    config = projector.ProjectorConfig()
    embed= config.embeddings.add()
    embed.tensor_name = 'embedding:0'
    embed.metadata_path = 'metadata.tsv'

    projector.visualize_embeddings(writer, config)

    saver.save(sess, os.path.join(
        FLAGS.logdir, 'projector/model.ckpt'), global_step=1)
    with open(FLAGS.logdir + "/projector/metadata.tsv", 'w') as f:
        nloop = FLAGS.nmax
        if tsv_data is not None:
            nloop += 1
        for i in range(nloop):
            if tsv_data is None: s = '_'+str(i)
            else: s = tsv_data[i]
            f.write("%s\n" % s)

def main(_):
    global FLAGS
    if tf.gfile.Exists(FLAGS.logdir + '/projector'):
        tf.gfile.DeleteRecursively(FLAGS.logdir + '/projector')
        tf.gfile.MkDir(FLAGS.logdir + '/projector')
    tf.gfile.MakeDirs(FLAGS.logdir  + '/projector')
    embeddingShow()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='input numpy data')
    parser.add_argument('--tsv', type=str, default='', help='tsv file for labels')
    parser.add_argument('--logdir', type=str, default='summary', help='Summaries log dir')
    parser.add_argument('--nmax', type=int, default=1000, help='max size of show')
    parser.add_argument('--random', type=int, default=1, help='max size of show')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)

