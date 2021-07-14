#!/usr/bin/env python3

import os
import argparse

import utils.np_utils as npu
import numpy as np
import tensorflow as tf

import net
import utils.utils as ut
import utils.tf_utils as tfu
from utils.dataset import Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
opts = parser.parse_args()

TLIST = 'data/train.txt'
VPATH = 'data/valset2'

BSZ = 1
IMSZ = 448
LR = 1e-4
DROP = (1.1e6, 1.25e6) # Learning rate drop
MAXITER = 1.5e6

VALFREQ = 2e1
SAVEFREQ = 5e4

wts = 'wts'
if not os.path.exists(wts):
    os.makedirs(wts)
model = net.Net(ksz=15, num_basis=90, burst_length=2)


def get_lr(niter):
    if niter < DROP[0]:
        return LR
    elif niter >= DROP[0] and niter < DROP[1]:
        return LR / np.sqrt(10.)
    else:
        return LR / 10.

#########################################################################

# Check for saved weights & optimizer states
msave = ut.ckpter(wts + '/iter_*.model.npz')
ssave = ut.ckpter(wts + '/iter_*.state.npz')
ut.logopen(wts + '/train.log')
niter = msave.iter


with tf.device('/cpu:0'):
    global_step = tf.placeholder(dtype=tf.int64, shape=[])

    # Set up optimizer
    lr = tf.placeholder(shape=[], dtype=tf.float32)
    opt = tf.train.AdamOptimizer(lr)

    # Data loading setup
    dataset = Dataset(TLIST, VPATH, bsz=BSZ, psz=IMSZ,
                      ngpus=opts.ngpus, nthreads=4 * opts.ngpus)

    # Calculate grads for each tower
    tower_grads = []
    tower_loss, tower_lvals = [], []
    for i in range(opts.ngpus):
        with tf.device('/gpu:%d' % i):
            example = dataset.batches[i]

            alpha = example['alpha'][:, None, None, None]
            dimmed_ambient, _ = tfu.dim_image(
                example['ambient'], alpha=alpha)
            dimmed_warped_ambient, _ = tfu.dim_image(
                example['warped_ambient'], alpha=alpha)

            # Make the flash brighter by increasing the brightness of the
            # flash-only image.
            flash = example['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
            warped_flash = example['warped_flash_only'] * \
                ut.FLASH_STRENGTH + dimmed_warped_ambient

            sig_read = example['sig_read'][:, None, None, None]
            sig_shot = example['sig_shot'][:, None, None, None]
            noisy_ambient, _, _ = tfu.add_read_shot_noise(
                dimmed_ambient, sig_read=sig_read, sig_shot=sig_shot)
            noisy_flash, _, _ = tfu.add_read_shot_noise(
                warped_flash, sig_read=sig_read, sig_shot=sig_shot)

            noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
            noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)
            net_input = tf.concat([noisy, noise_std], axis=-1)

            denoise = model.forward(net_input) / alpha

            denoise = tfu.camera_to_rgb(
                denoise, example['color_matrix'], example['adapt_matrix'])
            ambient = tfu.camera_to_rgb(
                example['ambient'],
                example['color_matrix'], example['adapt_matrix'])

            # Loss
            l2_loss = tfu.l2_loss(denoise, ambient)
            gradient_loss = tfu.gradient_loss(denoise, ambient)
            psnr = tfu.get_psnr(denoise, ambient)

            loss = l2_loss + gradient_loss
            lvals = [loss, l2_loss, gradient_loss, psnr]
            lnms = ['loss', 'l2_pixel', 'l1_gradient', 'psnr']

            tower_loss.append(loss)
            tower_lvals.append(lvals)

            grads = opt.compute_gradients(
                loss, var_list=list(model.weights.values()))
            tower_grads.append(grads)

    # Update step
    with tf.device('/gpu:0'):
        grads = ut.average_gradients(tower_grads)
        tStep = opt.apply_gradients(grads)

        # Aggregate losses for output
        loss = tf.reduce_mean(tf.stack(tower_loss, axis=0), axis=0)
        lvals = tf.reduce_mean(tf.stack(tower_lvals, axis=0), axis=0)
        tnms = [l + '.t' for l in lnms]
        vnms = [l + '.v' for l in lnms]

#########################################################################
# Start TF session
sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
dataset.init_handles(sess)

#########################################################################
# Load saved weights if any
if niter > 0:
    mfn = wts + "/iter_%06d.model.npz" % niter
    sfn = wts + "/iter_%06d.state.npz" % niter

    ut.mprint("Restoring model from " + mfn)
    ut.loadNet(mfn, model.weights, sess)
    ut.mprint("Restoring state from " + sfn)
    ut.loadAdam(sfn, opt, model.weights, sess)
    ut.mprint("Done!")

#########################################################################
# Main Training loop

stop = False
ut.mprint("Starting from Iteration %d" % niter)
dataset.swap_train(sess)

while niter < MAXITER and not ut.stop:

    # Validate model every so often
    if niter % VALFREQ == 0 and niter != 0:
        ut.mprint("Validating model")
        dataset.swap_val(sess)
        vloss = []
        while True:
            try:
                outs = sess.run(lvals, feed_dict={global_step: niter})
                vloss.append(outs)
            except tf.errors.OutOfRangeError:
                break
        vloss = np.mean(np.stack(vloss, axis=0), axis=0)
        ut.vprint(niter, vnms, vloss.tolist())

        dataset.swap_train(sess)

    # Run training step and print losses
    if niter % 100 == 0:
        outs = sess.run(
            [lvals, tStep],
            feed_dict={lr: get_lr(niter), global_step: niter}
        )
        ut.vprint(niter, tnms, outs[0].tolist())
        ut.vprint(niter, ['lr'], [get_lr(niter)])
    else:
        outs = sess.run(
            [loss, tStep],
            feed_dict={lr: get_lr(niter), global_step: niter}
        )
        ut.vprint(niter, ['loss.t'], [outs[0]])

    niter = niter + opts.ngpus

    # Save model weights if needed
    if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
        mfn = wts + "/iter_%06d.model.npz" % niter
        sfn = wts + "/iter_%06d.state.npz" % niter

        ut.mprint("Saving model to " + mfn)
        ut.saveNet(mfn, model.weights, sess)
        ut.mprint("Saving state to " + sfn)
        ut.saveAdam(sfn, opt, model.weights, sess)
        ut.mprint("Done!")
        msave.clean(every=SAVEFREQ, last=1)
        ssave.clean(every=SAVEFREQ, last=1)


# Save last
if msave.iter < niter:
    mfn = wts + "/iter_%06d.model.npz" % niter
    sfn = wts + "/iter_%06d.state.npz" % niter

    ut.mprint("Saving model to " + mfn)
    ut.saveNet(mfn, model.weights, sess)
    ut.mprint("Saving state to " + sfn)
    ut.saveAdam(sfn, opt, model.weights, sess)
    ut.mprint("Done!")
    msave.clean(every=SAVEFREQ, last=1)
    ssave.clean(every=SAVEFREQ, last=1)
