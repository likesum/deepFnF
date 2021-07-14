'''Preprocess exifs data of the flahs/no-flash dataset'''
import glob
import argparse

import exiftool
from six.moves import cPickle as pkl
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='directory of dataset')
opts = parser.parse_args()


ILLUMINANTS = {
    "a": (1.09850, 1.00000, 0.35585),
    "b": (0.99072, 1.00000, 0.85223),
    "c": (0.98074, 1.00000, 1.18232),
    "d50": (0.96422, 1.00000, 0.82521),
    "d55": (0.95682, 1.00000, 0.92149),
    "d65": (0.95047, 1.00000, 1.08883),
    "d75": (0.94972, 1.00000, 1.22638),
    "e": (1.00000, 1.00000, 1.00000),
    "f2": (0.99186, 1.00000, 0.67393),
    "f7": (0.95041, 1.00000, 1.08747),
    "f11": (1.00962, 1.00000, 0.64350)
}

# Chromatic Adaptation Matrices
# http://brucelindbloom.com/Eqn_ChromAdapt.html
# bradford
ADAPTATION_MATRIX = np.array(
    (
        (0.8951, 0.2664, -0.1614),
        (-0.7502, 1.7135, 0.0367),
        (0.0389, -0.0685, 1.0296),
    )
)

CONVERSION_MATRICES = {
    "xyz_to_rgb": np.array(
        (
            (3.24071, -1.53726, -0.498571),
            (-0.969258, 1.87599, 0.0415557),
            (0.0556352, -0.203996, 1.05707),
        )
    ),
    "rgb_to_xyz": np.array(
        (
            (0.412424, 0.357579, 0.180464),
            (0.212656, 0.715158, 0.0721856),
            (0.0193324, 0.119193, 0.950444),
        )
    ),
}

ILLUMINANT_CODE = {
    17: 'a',
    19: 'c',
    20: 'd55',
    21: 'd65',
    23: 'd50'
}


def _get_adaptation_matrix(illuminant):
    wp_src = ILLUMINANTS[illuminant]
    wp_dst = ILLUMINANTS['d65']
    rgb_src = np.dot(ADAPTATION_MATRIX, wp_src)
    rgb_dst = np.dot(ADAPTATION_MATRIX, wp_dst)
    m_rat = np.diag(rgb_dst / rgb_src)
    m_xfm = np.dot(np.dot(np.linalg.pinv(
        ADAPTATION_MATRIX), m_rat), ADAPTATION_MATRIX)
    return m_xfm


def decode_illuminant(illuminant):
    if illuminant in ILLUMINANT_CODE:
        return ILLUMINANT_CODE[illuminant]
    else:
        return 'd65'


data = {}
nms = glob.glob('%s/*.png' % opts.data_dir)

with exiftool.ExifTool() as et:
    for nm in nms:
        if 'ambient' in nm:
            metadata = et.get_metadata(nm)
            color_matrix = [float(i)
                            for i in metadata['PNG:Comment'].split('     ')]
            color_matrix = np.reshape(np.array(color_matrix), (3, 3))
            illuminant = int(metadata['PNG:Description'])
            illum_adapt_mat = _get_adaptation_matrix(
                decode_illuminant(illuminant))
            data[nm.replace('_ambient.png', '')] = (
                color_matrix, illum_adapt_mat)

with open('data/exifs.pkl', 'wb') as f:
    pkl.dump(data, f, protocol=2)
