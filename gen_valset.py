#!/usr/bin/env python3

import os

import numpy as np
import tensorflow as tf

from utils.dataset import Dataset

outpath = 'data/valset'
if not os.path.exists(outpath):
    os.makedirs(outpath)

TLIST = 'data/train.txt'
VLIST = 'data/val.txt'

def gamma(img):
    return img**(1/2.2)

BSZ = 1
IMSZ = 448

dataset = Dataset(TLIST, VLIST, bsz=BSZ, psz=IMSZ, onfly_val=True)
example = dataset.batches[0]

sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
dataset.init_handles(sess)
dataset.swap_val(sess)

c = 0
while True:
    try:
        data = sess.run(example)
        np.savez('%s/%d.npz' % (outpath, c), **data)
        c += 1
    except tf.errors.OutOfRangeError:
        break





