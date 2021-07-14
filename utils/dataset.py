import glob
import functools

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pkl
from utils.tf_spatial_transformer import transformer

with open('data/exifs.pkl', 'rb') as f:
    COLOR_MAP_DATA = pkl.load(f)

DATA_NAMES = [
    'ambient', 'warped_ambient', 'flash_only', 'warped_flash_only',
    'color_matrix', 'adapt_matrix', 'alpha', 'sig_read', 'sig_shot',
]


def load_image(filename, color_matrix, adapt_matrix):
    '''Load image and its camera matrices'''
    example = {}
    ambient = tf.io.read_file(filename + '_ambient.png')
    ambient = tf.image.decode_png(ambient, channels=3, dtype=tf.uint16)
    example['ambient'] = tf.cast(ambient, tf.float32) / 65535.

    flash_only = tf.io.read_file(filename + '_flash.png')
    flash_only = tf.image.decode_png(flash_only, channels=3, dtype=tf.uint16)
    example['flash_only'] = tf.cast(flash_only, tf.float32) / 65535.

    example['color_matrix'] = color_matrix
    example['adapt_matrix'] = adapt_matrix
    return example


def gen_homography(
        example, jitter, min_scale, max_scale, theta, psz, is_val=False):
    '''Randomly warp the image'''
    ambient = tf.clip_by_value(example['ambient'], 0., 1.)
    flash_only = tf.clip_by_value(example['flash_only'], 0., 1.)
    height, width = tf.shape(ambient)[0], tf.shape(ambient)[1]

    valid = int(jitter / min_scale)
    v_error = tf.maximum((psz + 2 * valid - height + 1) // 2, 0)
    h_error = tf.maximum((psz + 2 * valid - width + 1) // 2, 0)
    ambient = tf.pad(ambient, [[v_error, v_error], [h_error, h_error], [0, 0]])
    flash_only = tf.pad(
        flash_only, [[v_error, v_error], [h_error, h_error], [0, 0]])
    height = height + 2 * v_error
    width = width + 2 * h_error

    if not is_val:
        y = tf.random.uniform([], valid, tf.shape(
            ambient)[0] - valid - psz + 1, tf.int32)
        x = tf.random.uniform([], valid, tf.shape(
            ambient)[1] - valid - psz + 1, tf.int32)
    else:
        y = valid
        x = valid

    fov = np.deg2rad(90)
    f = psz / 2 / np.tan(fov / 2.)
    intrinsic = tf.convert_to_tensor([
        [f, 0, tf.cast(x, tf.float32) + psz / 2.],
        [0, f, tf.cast(y, tf.float32) + psz / 2.],
        [0, 0, 1]])
    intrinsic_inv = tf.linalg.inv(intrinsic)

    curr = tf.eye(3)
    scale = tf.random.uniform([], min_scale, max_scale)
    theta_x = tf.random.uniform([], -theta, theta)
    theta_y = tf.random.uniform([], -theta, theta)
    theta_z = tf.random.uniform([], -theta, theta)
    shift_x = tf.random.uniform([], -jitter, jitter)
    shift_y = tf.random.uniform([], -jitter, jitter)

    rotate_x = tf.convert_to_tensor([
        [1, 0, 0],
        [0, tf.cos(theta_x), -tf.sin(theta_x)],
        [0, tf.sin(theta_x), tf.cos(theta_x)]])
    rotate_y = tf.convert_to_tensor([
        [tf.cos(theta_y), 0, -tf.sin(theta_y)],
        [0, 1, 0],
        [tf.sin(theta_y), 0, tf.cos(theta_y)]])
    rotate_z = tf.convert_to_tensor([
        [tf.cos(theta_z), -tf.sin(theta_z), 0],
        [tf.sin(theta_z), tf.cos(theta_z), 0],
        [0, 0, 1]])
    rotate = tf.matmul(tf.matmul(rotate_x, rotate_y), rotate_z)
    rotate_homo = tf.matmul(tf.matmul(intrinsic, rotate), intrinsic_inv)

    scale_shift = tf.convert_to_tensor(
        [[scale, 0, -shift_x], [0, scale, -shift_y], [0, 0, 1]])

    H = tf.matmul(rotate_homo, scale_shift)
    H = tf.matmul(H, curr)
    H = tf.reshape(H, [1, 9])

    warped_flash_only, _ = transformer(flash_only[None], H, [height, width])
    warped_flash_only = tf.squeeze(warped_flash_only, axis=0)
    warped_flash_only = warped_flash_only[y:y + psz, x:x + psz, :]
    # due to numerical issue, might be values that are slightly larger than 1.0
    example['warped_flash_only'] = tf.clip_by_value(warped_flash_only, 0., 1.)

    warped_ambient, _ = transformer(ambient[None], H, [height, width])
    warped_ambient = tf.squeeze(warped_ambient, axis=0)
    warped_ambient = warped_ambient[y:y + psz, x:x + psz, :]
    example['warped_ambient'] = tf.clip_by_value(warped_ambient, 0., 1.)

    example['ambient'] = ambient[y:y + psz, x:x + psz, :]
    example['flash_only'] = flash_only[y:y + psz, x:x + psz, :]

    return example


def gen_random_params(
        example, min_alpha=0.02, max_alpha=0.2,
        min_read=-3., max_read=-2, min_shot=-2., max_shot=-1.3):
    '''Random noise parameters'''
    example['alpha'] = tf.pow(
        10., tf.random.uniform([], np.log10(min_alpha), np.log10(max_alpha)))
    example['sig_read'] = tf.pow(
        10., tf.random.uniform([], min_read, max_read))
    example['sig_shot'] = tf.pow(
        10., tf.random.uniform([], min_shot, max_shot))
    return example


def valset_generator(data_path):
    for i in range(256):
        data = np.load('%s/%d.npz' % (data_path, i))
        example = {}
        for name in DATA_NAMES:
            example[name] = np.squeeze(data[name])

        yield example


class Dataset:
    def __init__(
            self, train_list, val_path,
            bsz=32, psz=512, jitter=2,
            min_scale=0.98, max_scale=1.02, theta=np.deg2rad(0.5),
            ngpus=1, nthreads=4, onfly_val=False):
        self.train = TrainSet(
            train_list, bsz, psz, jitter,
            min_scale, max_scale, theta, ngpus, nthreads)
        if onfly_val:
            self.val = _OnFlyValSet(
                val_path, bsz, psz, jitter, min_scale, 
                max_scale, theta, ngpus, nthreads)
        else:
            self.val = ValSet(val_path, bsz, ngpus)

        self._handle = tf.Variable(
            tf.zeros(shape=[], dtype=tf.string), trainable=False)
        self._handle_ph = tf.placeholder(tf.string, shape=[])
        self._handle_assign_op = tf.assign(self._handle, self._handle_ph).op

        self.iterator = tf.data.Iterator.from_string_handle(
            self._handle, self.train.output_types, self.train.output_shapes)

        # One batch for each gpu
        self.batches = []
        for i in range(ngpus):
            example = self.iterator.get_next()
            for name, data in example.items():
                if name in ['ambient', 'warped_ambient', 'flash_only', 'warped_flash_only']:
                    data.set_shape([bsz, psz, psz, 3])
            self.batches.append(example)

    def init_handles(self, sess):
        self.train_handle = self.train.get_handle(sess)
        self.val_handle = self.val.get_handle(sess)

    def swap_train(self, sess):
        sess.run(self._handle_assign_op, feed_dict={
                 self._handle_ph: self.train_handle})

    def swap_val(self, sess):
        self.val.initialize(sess)
        sess.run(self._handle_assign_op, feed_dict={
                 self._handle_ph: self.val_handle})


class TrainSet:
    def __init__(
            self, file_list, bsz, psz, jitter,
            min_scale, max_scale, theta, ngpus, nthreads):
        files = [l.strip() for l in open(file_list)]

        gen_homography_fn = functools.partial(
            gen_homography, jitter=jitter, min_scale=min_scale,
            max_scale=max_scale, theta=theta, psz=psz)

        color_matrices = np.stack(
            [COLOR_MAP_DATA[nm][0] for nm in files],
            axis=0).astype(np.float32)
        adapt_matrices = np.stack(
            [COLOR_MAP_DATA[nm][1] for nm in files],
            axis=0).astype(np.float32)

        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(files),
            tf.data.Dataset.from_tensor_slices(color_matrices),
            tf.data.Dataset.from_tensor_slices(adapt_matrices)
        ))
        self.dataset = (dataset
                        .repeat()
                        .shuffle(buffer_size=len(files))
                        .map(load_image, num_parallel_calls=nthreads)
                        .map(gen_homography_fn, num_parallel_calls=nthreads)
                        .map(gen_random_params, num_parallel_calls=nthreads)
                        .batch(bsz)
                        .prefetch(ngpus)
                        )
        self.iterator = self.dataset.make_one_shot_iterator()

        self.output_types = self.dataset.output_types
        self.output_shapes = self.dataset.output_shapes

    def get_handle(self, sess):
        return sess.run(self.iterator.string_handle())


class ValSet:
    def __init__(self, val_path, bsz, ngpus):
        generator = functools.partial(valset_generator, data_path=val_path)
        dataset = tf.data.Dataset.from_generator(
            generator,
            {name: tf.float32 for name in DATA_NAMES})

        self.dataset = (dataset
                        .batch(bsz, drop_remainder=True)
                        .prefetch(ngpus)
                        )
        self.iterator = self.dataset.make_initializable_iterator()

    def initialize(self, sess):
        sess.run(self.iterator.initializer)

    def get_handle(self, sess):
        return sess.run(self.iterator.string_handle())


class _OnFlyValSet:
    def __init__(
            self, file_list, bsz, psz, jitter,
            min_scale, max_scale, theta, ngpus, nthreads):
        files = [l.strip() for l in open(file_list)]

        gen_homography_fn = functools.partial(
            gen_homography, jitter=jitter, min_scale=min_scale,
            max_scale=max_scale, theta=theta, psz=psz, is_val=True)

        color_matrices = np.stack(
            [COLOR_MAP_DATA[nm][0] for nm in files],
            axis=0).astype(np.float32)
        adapt_matrices = np.stack(
            [COLOR_MAP_DATA[nm][1] for nm in files],
            axis=0).astype(np.float32)

        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(files),
            tf.data.Dataset.from_tensor_slices(color_matrices),
            tf.data.Dataset.from_tensor_slices(adapt_matrices)
        ))
        self.dataset = (dataset
                        .map(load_image, num_parallel_calls=nthreads)
                        .map(gen_homography_fn, num_parallel_calls=nthreads)
                        .map(gen_random_params, num_parallel_calls=nthreads)
                        .batch(bsz, drop_remainder=True)
                        .prefetch(ngpus)
                        )
        self.iterator = self.dataset.make_initializable_iterator()

    def initialize(self, sess):
        sess.run(self.iterator.initializer)

    def get_handle(self, sess):
        return sess.run(self.iterator.string_handle())
