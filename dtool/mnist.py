#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright: 2015 Jingyi Xiao
# FileName: mnist.py
# Date: 2019 2019年06月10日 14:50:00
# Encoding: utf-8
# Author: Jingyi Xiao <kxwarning@126.com>
# Note: This source file is NOT a freeware

__author__="Jingyi"

import os, sys, time
import argparse
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, default='datas', help='output numpy data')
FLAGS, unparsed = parser.parse_known_args()

def main():
    mnist = input_data.read_data_sets('tmp/', one_hot=False)
    print(dir(mnist))
    print(dir(mnist.train))
    X = mnist.train.images
    y = mnist.train.labels
    print(X, y)
    
    tf.gfile.MakeDirs(FLAGS.out)
    np.save(FLAGS.out + '/x.npy', X)
    with open(FLAGS.out + '/metadata.tsv', 'w') as f:
        yls = y.tolist()
        f.write("Index\tLabel\n")
        #yls = [str(i) + '\t' + str(one) for i, one in enumerate(yls)]
        yls = ["_%d\t_%d" % (i, one) for i, one in enumerate(yls)]
        f.write("\n".join(yls))
    return

if __name__ == "__main__":
    main()

# Modeline for ViM {{{
# vim:set ts=4:
# vim600:fdm=marker fdl=0 fdc=3:
# }}}

