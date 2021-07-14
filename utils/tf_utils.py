import numpy as np
import tensorflow as tf
import time

CONVERSION_MATRICES = {
    "xyz_to_rgb": np.array(
        (
            (3.24071, -1.53726, -0.498571),
            (-0.969258, 1.87599, 0.0415557),
            (0.0556352, -0.203996, 1.05707),
        ), dtype=np.float32
    ),
    "rgb_to_xyz": np.array(
        (
            (0.412424, 0.357579, 0.180464),
            (0.212656, 0.715158, 0.0721856),
            (0.0193324, 0.119193, 0.950444),
        ), dtype=np.float32
    ),
}


def dim_image(
        ambient, min_alpha=0.02, max_alpha=0.2, alpha=None):
    bsz = tf.shape(ambient)[0]
    if alpha is None:
        alpha = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], np.log10(min_alpha), np.log10(max_alpha)))
    return alpha * ambient, alpha


def add_read_shot_noise(
    imgs, sig_read=None, sig_shot=None, 
    min_read=-3., max_read=-2, min_shot=-2., max_shot=-1.3):
    if sig_read is None or sig_shot is None:
        bsz = tf.shape(imgs)[0]
        sig_read = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], min_read, max_read))
        sig_shot = tf.pow(10., tf.random.uniform(
            [bsz, 1, 1, 1], min_shot, max_shot))
    read = sig_read * tf.random.normal(tf.shape(imgs))
    shot = tf.sqrt(imgs) * sig_shot * tf.random.normal(tf.shape(imgs))
    noisy = imgs + shot + read
    return noisy, sig_read, sig_shot


def estimate_std(noisy, sig_read, sig_shot):
    return tf.sqrt(sig_read**2 + tf.maximum(0., noisy) * sig_shot**2)


def reverse_gamma(imgs, gamma=1. / 2.2):
    return imgs**(1. / gamma)


def gamma_correct(x):
    b = .0031308
    gamma = 1. / 2.4
    a = .055
    k0 = 12.92
    def gammafn(x): return (1 + a) * tf.pow(tf.maximum(x, b), gamma) - a
    srgb = tf.where(x < b, k0 * x, gammafn(x))
    k1 = (1 + a) * gamma
    srgb = tf.where(x > 1, k1 * x - k1 + 1, srgb)
    return srgb


def camera_to_rgb(imgs, color_matrix, adapt_matrix):
    b, c = tf.shape(imgs)[0], tf.shape(imgs)[-1]
    imsp = tf.shape(imgs)
    imgs = tf.reshape(imgs, [b, -1, c])
    imgs = tf.transpose(imgs, [0, 2, 1])

    xyz = tf.linalg.solve(color_matrix, imgs)
    xyz = tf.linalg.matmul(adapt_matrix, xyz)
    rgb = tf.linalg.matmul(CONVERSION_MATRICES['xyz_to_rgb'], xyz)
    rgb = gamma_correct(rgb)

    rgb = tf.transpose(rgb, [0, 2, 1])
    rgb = tf.reshape(rgb, imsp)
    return rgb


def get_gradient(imgs):
    return tf.concat([
        .5 * (imgs[:, 1:, :-1, :] - imgs[:, :-1, :-1, :]),
        .5 * (imgs[:, :-1, 1:, :] - imgs[:, :-1, :-1, :])], axis=-1)


def gradient_loss(pred, gt):
    return l1_loss(get_gradient(pred), get_gradient(gt))


def l1_loss(pred, gt):
    return tf.reduce_mean(tf.abs(pred - gt))


def l2_loss(pred, gt):
    return tf.reduce_mean(tf.square(pred - gt))


def get_psnr(pred, gt):
    pred = tf.clip_by_value(pred, 0., 1.)
    gt = tf.clip_by_value(gt, 0., 1.)
    mse = tf.reduce_mean((pred - gt)**2.0, axis=[1, 2, 3])
    psnr = tf.reduce_mean(-10. * tf.log(mse) / tf.log(10.))
    return psnr


def apply_filtering(imgs, kernels):
    b, h, w, c = imgs.get_shape().as_list()
    burst_length = c // 3
    b = tf.shape(imgs)[0]
    ksz = int(np.sqrt(kernels.get_shape().as_list()[-1] / burst_length / 3))
    padding = (ksz - 1) // 2
    imgs = tf.pad(imgs, [[0, 0], [padding, padding], [
                  padding, padding], [0, 0]], 'REFLECT')
    patches = tf.extract_image_patches(
        imgs, [1, ksz, ksz, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
    patches = tf.reshape(patches, [b, h, w, ksz * ksz, burst_length * 3])
    kernels = tf.reshape(kernels, [b, h, w, ksz * ksz, burst_length * 3])
    framewise = tf.reduce_sum(patches * kernels, axis=-2)
    framewise = tf.reshape(framewise, [b, h, w, burst_length, 3])
    out = tf.reduce_sum(framewise, axis=-2)
    return out


def apply_dilated_filtering(imgs, kernels, dilation=1):
    b, h, w, c = imgs.get_shape().as_list()
    burst_length = c // 3
    b = tf.shape(imgs)[0]
    ksz = int(np.sqrt(kernels.get_shape().as_list()[-1] / burst_length / 3))
    padding = (ksz - 1) * dilation // 2
    imgs = tf.pad(
        imgs, [[0, 0], [padding, padding],
        [padding, padding], [0, 0]], 'REFLECT')
    patches = tf.extract_image_patches(
        imgs, [1, ksz, ksz, 1], [1, 1, 1, 1],
        [1, dilation, dilation, 1], 'VALID')
    patches = tf.reshape(patches, [b, h, w, ksz * ksz, burst_length * 3])
    kernels = tf.reshape(kernels, [b, h, w, ksz * ksz, burst_length * 3])
    framewise = tf.reduce_sum(patches * kernels, axis=-2)
    framewise = tf.reshape(framewise, [b, h, w, burst_length, 3])
    out = tf.reduce_sum(framewise, axis=-2)
    return out


def bilinear_filter(image, ksz):
    if ksz == 3:
        kernel = np.array([0.5, 1., 0.5], dtype=np.float32).reshape([3, 1])
    elif ksz == 7:
        kernel = np.array([0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25],
                          dtype=np.float32).reshape([7, 1])
    kernel = np.matmul(kernel, kernel.T)
    kernel = kernel / np.sum(kernel)
    kernel = tf.tile(kernel[..., None, None], [1, 1, tf.shape(image)[-1], 1])
    image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
    return image
