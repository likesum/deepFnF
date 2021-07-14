from collections import OrderedDict

import numpy as np
import tensorflow as tf

import utils.tf_utils as tfu


class Net:
    def __init__(self, num_basis=90, ksz=15, burst_length=2):
        self.weights = {}
        self.activations = OrderedDict()
        self.num_basis = num_basis
        self.ksz = ksz
        self.burst_length = burst_length

    def conv(
            self, name, inp, outch, ksz=3,
            stride=1, relu=True, pad='SAME', activation_name=None):
        '''Wrapper of conv'''
        inch = inp.get_shape().as_list()[-1]
        ksz = [ksz, ksz, inch, outch]

        wnm = name + "_w"
        if wnm in self.weights.keys():
            w = self.weights[wnm]
        else:
            sq = np.sqrt(3.0 / np.float32(ksz[0] * ksz[1] * ksz[2]))
            w = tf.Variable(tf.random_uniform(
                ksz, minval=-sq, maxval=sq, dtype=tf.float32))
            self.weights[wnm] = w

        out = tf.nn.conv2d(inp, w, [1, stride, stride, 1], pad)

        bnm = name + "_b"
        if bnm in self.weights.keys():
            b = self.weights[bnm]
        else:
            b = tf.Variable(tf.constant(0, shape=[ksz[-1]], dtype=tf.float32))
            self.weights[bnm] = b
        out = out + b

        if relu:
            out = tf.nn.relu(out)

        if activation_name is not None:
            self.activations[activation_name] = out

        return out

    def down_block(self, out, nch, pfx=''):
        '''
        Downsampling block, including two conv layers w/ one maxpooling layer
        Args:
            out: output from previous layer
            nch: number of channels for the block
            pfx: prefix of names for layers in this block
        Return:
            down: output of the block after downsampling
            out: output of the block right before downsampling
        '''
        out = self.conv(pfx + '_1', out, nch, ksz=3, stride=1)
        out = self.conv(pfx + '_2', out, nch, ksz=3, stride=1)
        down = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        self.activations[pfx] = down
        self.activations['skip_' + pfx] = out
        return down, out

    def up_block(self, out, nch, skip, pfx=''):
        '''
        Upsampling block, including:
            one upsampling with bilinear resizing,
            one conv layer,
            skip connection
            two more conv layers
        Args:
            out: output from previous layer
            skip: output from the layer that is skip connected to this block
            nch: number of channels for the block
            pfx: prefix of names for layers in this block
        Return:
            out: output of this block
        '''
        out = tf.image.resize_bilinear(out, 2 * tf.shape(out)[1:3])
        out = self.conv(pfx + '_1', out, nch, ksz=3, stride=1)

        out = tf.concat([out, skip], axis=-1)

        out = self.conv(pfx + '_2', out, nch, ksz=3, stride=1)
        out = self.conv(pfx + '_3', out, nch, ksz=3,
                        stride=1, activation_name=pfx)
        return out

    def kernel_up_block(self, out, nch, skip, pfx=''):
        '''
        Upsampling block, including:
            one upsampling with bilinear resizing,
            one conv layer,
            skip connection (with gloal average pooling)
            two more conv layers
        Args:
            out: output from previous layer
            skip: output from the layer that is skip connected to this block
            nch: number of channels for the block
            pfx: prefix of names for layers in this block
        Return:
            out: output of this block
        '''
        shape = tf.shape(out)
        out = tf.image.resize_bilinear(out, 2 * shape[1:3])
        out = self.conv(pfx + '_1', out, nch, ksz=3, stride=1)

        # resize the skip connection
        skip = tf.reduce_mean(skip, axis=[1, 2], keepdims=True)
        skip = tf.tile(skip, [1, 2 * shape[1], 2 * shape[2], 1])
        out = tf.concat([out, skip], axis=-1)

        out = self.conv(pfx + '_2', out, nch, ksz=3, stride=1)
        out = self.conv(pfx + '_3', out, nch, ksz=3,
                        stride=1, activation_name=pfx)
        return out


    def encode(self, out, pfx=''):
        out = self.conv(pfx + 'inp', out, 64)

        out, d1 = self.down_block(out, 64, pfx + 'down1')
        out, d2 = self.down_block(out, 128, pfx + 'down2')
        out, d3 = self.down_block(out, 256, pfx + 'down3')
        out, d4 = self.down_block(out, 512, pfx + 'down4')
        out, d5 = self.down_block(out, 1024, pfx + 'down5')

        out = self.conv(pfx + 'bottleneck_1', out, 1024)
        out = self.conv(pfx + 'bottleneck_2', out, 1024,
                        activation_name=pfx + 'bottleneck')
        return out, [d1, d2, d3, d4, d5]

    def decode(self, out, skips, pfx=''):
        d1, d2, d3, d4, d5 = skips
        out = self.up_block(out, 512, d5, pfx + 'up1')
        out = self.up_block(out, 256, d4, pfx + 'up2')
        out = self.up_block(out, 128, d3, pfx + 'up3')
        out = self.up_block(out, 64, d2, pfx + 'up4')
        out = self.up_block(out, 64, d1, pfx + 'up5')

        out = self.conv(pfx + 'end_1', out, 64)
        out = self.conv(pfx + 'end_2', out, 64, activation_name=pfx + 'end')

        return out

    def create_basis(self):
        '''Predict image-specific basis'''
        assert self.ksz == 15
        bottleneck = self.activations['bottleneck']
        out = tf.reduce_mean(bottleneck, axis=[1, 2], keepdims=True)  # 1x1
        out = self.kernel_up_block(
            out, 512, self.activations['skip_down5'], 'k_up1')  # 2x2
        out = self.kernel_up_block(
            out, 256, self.activations['skip_down4'], 'k_up2')  # 4x4
        out = self.kernel_up_block(
            out, 256, self.activations['skip_down3'], 'k_up3')  # 8x8
        out = self.kernel_up_block(
            out, 128, self.activations['skip_down2'], 'k_up4')  # 16x16
        out = self.conv('k_conv', out, 128, ksz=2, stride=1, pad='VALID')
        out = self.conv('k_output_1', out, 128)
        out = self.conv('k_output_2', out, 3 * 2 * self.num_basis, relu=False)
        out = tf.reshape(
            out, [-1, self.ksz * self.ksz * 3 * 2, self.num_basis])
        self.basis = tf.transpose(out, [0, 2, 1])

    def predict_coeff(self, inp):
        '''Predict per-pixel coefficient vector given the input'''
        self.imsp = tf.shape(inp)

        out, skips = self.encode(inp)
        out = self.decode(out, skips)
        out = self.conv('output', out, self.num_basis + 3, relu=False)
        self.coeffs_pre_soft = out
        self.coeffs = out[..., :self.num_basis]
        self.scale = out[..., -3:]
        self.activations['output'] = self.coeffs

    def combine(self):
        '''Combine coeffs and basis to get a per-pixel kernel'''
        imsp = self.imsp
        coeffs = tf.reshape(
            self.coeffs, [-1, imsp[1] * imsp[2], self.num_basis])
        self.kernels = tf.matmul(
            coeffs,
            self.basis
        )  # (h * w) x (ksz * ksz * 3 * 2)
        self.kernels = tf.reshape(
            self.kernels, [-1, imsp[1], imsp[2], self.ksz * self.ksz * 3, 2])
        self.activations['decoding'] = self.kernels

    def forward(self, inp):
        self.predict_coeff(inp)
        self.create_basis()
        self.combine()

        filtered_ambient = tfu.apply_filtering(
            inp[:, :, :, :3], self.kernels[..., 0])

        # "Bilinearly upsample kernels + filtering"
        # is equivalent to
        # "filter the image with a bilinear kernel + dilated filter the image
        # with the original kernel".
        # This will save more memory.
        smoothed_ambient = tfu.bilinear_filter(inp[:, :, :, :3], ksz=7)
        smoothed_ambient = tfu.apply_dilated_filtering(
            smoothed_ambient, self.kernels[..., 1], dilation=4)
        filtered_ambient = filtered_ambient + smoothed_ambient
        denoised = filtered_ambient * self.scale

        return denoised
