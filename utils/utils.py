import re
import os
import time
import sys
import signal
from glob import glob
import numpy as np
import tensorflow as tf

from imageio import imread, imsave
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

FLASH_STRENGTH = 2.

# Logic for trapping ctrlc and setting stop
stop = False
_orig = None


def handler(a, b):
    global stop
    stop = True
    signal.signal(signal.SIGINT, _orig)


_orig = signal.signal(signal.SIGINT, handler)


# Log print
_log = None


def logopen(fn):
    global _log
    _log = open(fn, 'a')


def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()
    _log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    _log.flush()


def vprint(it, nms, vals):
    s = '[%06d]' % it
    for i in range(len(nms)):
        s = s + ' ' + nms[i] + ' = %.3e' % vals[i]
    mprint(s)

# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest


class ckpter:
    def __init__(self, wcard):
        self.wcard = wcard
        self.load()

    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst = [(l, int(re.match('.*/.*_(\d+)', l).group(1)))
                   for l in lst]
            self.lst = sorted(lst, key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst = []
            self.iter = 0
            self.latest = None

    def clean(self, every=0, last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])


# Save/load networks
def saveNet(fn, dic, sess):
    wts = sess.run(dic)
    np.savez(fn, **wts)


def loadNet(fn, dic, sess):
    wts = np.load(fn)
    ops, fd = [], {}
    for k in dic.keys():
        ph = tf.placeholder(tf.float32)
        ops.append(dic[k].assign(ph).op)
        fd[ph] = wts[k]
    sess.run(ops, feed_dict=fd)


# Save/load Adam optimizer state
def saveAdam(fn, opt, vdict, sess):
    weights = {}
    beta1_power, beta2_power = opt._get_beta_accumulators()
    weights['b1p'] = beta1_power
    weights['b2p'] = beta2_power
    for nm in vdict.keys():
        v = vdict[nm]
        weights['m_%s' % nm] = opt.get_slot(v, 'm')
        weights['v_%s' % nm] = opt.get_slot(v, 'v')
    wts = sess.run(weights)
    np.savez(fn, **wts)


def loadAdam(fn, opt, vdict, sess):
    weights = np.load(fn)
    ph = tf.placeholder(tf.float32)
    beta1_power, beta2_power = opt._get_beta_accumulators()
    sess.run(beta1_power.assign(ph),
             feed_dict={ph: weights['b1p']})
    sess.run(beta2_power.assign(ph),
             feed_dict={ph: weights['b2p']})

    ops, fd = [], {}
    for nm in vdict.keys():
        v = vdict[nm]
        ph = tf.placeholder(tf.float32)
        ops.append(opt.get_slot(v, 'm').assign(ph).op)
        fd[ph] = weights['m_%s' % nm]

        ph = tf.placeholder(tf.float32)
        ops.append(opt.get_slot(v, 'v').assign(ph).op)
        fd[ph] = weights['v_%s' % nm]
    sess.run(ops, feed_dict=fd)


def average_gradients(tower_grads):
    '''
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    '''
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            grads.append(g)

        # Average over the 'tower' dimension.
        grad = tf.stack(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        # Return the first tower's pointer to the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
