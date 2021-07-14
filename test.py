#!/usr/bin/env python3

import os
import argparse

import utils.np_utils as npu
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import net
import utils.utils as ut
import utils.tf_utils as tfu

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--wts', default='wts/model.npz', help='path to trianed model')
opts = parser.parse_args()


def load_net(fn, model):
    wts = np.load(fn)
    for k, v in wts.items():
        model.weights[k] = tfe.Variable(v)


datapath = 'data/testset'
model_path = opts.wts

model = net.Net(ksz=15, num_basis=90, burst_length=2)

print("Restoring model from " + model_path)
load_net(model_path, model)
print('Done\n')

for k in range(4):
    metrics = []
    for c in range(128):
        data = np.load('%s/%d/%d.npz' % (datapath, k, c))

        alpha = data['alpha'][None, None, None, None]
        ambient = data['ambient']
        dimmed_ambient, _ = tfu.dim_image(data['ambient'], alpha=alpha)
        dimmed_warped_ambient, _ = tfu.dim_image(
            data['warped_ambient'], alpha=alpha)

        # Make the flash brighter by increasing the brightness of the
        # flash-only image.
        flash = data['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
        warped_flash = data['warped_flash_only'] * \
            ut.FLASH_STRENGTH + dimmed_warped_ambient

        noisy_ambient = data['noisy_ambient']
        noisy_flash = data['noisy_warped_flash']

        noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
        noise_std = tfu.estimate_std(
            noisy, data['sig_read'], data['sig_shot'])
        net_input = tf.concat([noisy, noise_std], axis=-1)

        denoise = model.forward(net_input)
        denoise = denoise / alpha

        ambient = tfu.camera_to_rgb(
            ambient, data['color_matrix'], data['adapt_matrix'])
        denoise = tfu.camera_to_rgb(
            denoise, data['color_matrix'], data['adapt_matrix'])

        ambient = np.clip(ambient, 0., 1.).squeeze()
        denoise = np.clip(denoise, 0., 1.).squeeze()

        mse = npu.get_mse(denoise, ambient)
        psnr = npu.get_psnr(denoise, ambient)
        ssim = npu.get_ssim(denoise, ambient)

        metrics.append([psnr, ssim])

    metrics = np.mean(metrics, axis=0).tolist()
    print('\nLevel %d' % (4 - k) +
          ': PSNR: %.3f, SSIM: %.4f' % tuple(metrics))

