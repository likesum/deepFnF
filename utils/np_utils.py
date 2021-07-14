from skimage.measure import compare_ssim
import numpy as np
import imageio

def imsave(nm, img):
    if len(img.shape) == 4:
        img = np.squeeze(img, 0)
    img = np.uint8(np.clip(img,0,1) * 255.)
    imageio.imsave(nm, img)


def get_mse(pred, gt):
    return np.mean(np.square(pred-gt))


def get_psnr(pred, gt):
    pred = pred.clip(0., 1.)
    gt = gt.clip(0., 1.)
    mse = np.mean((pred-gt)**2.0)
    psnr = -10. * np.log10(mse)
    return psnr


def get_ssim(pred, gt):
    ssim = compare_ssim(
        pred,
        gt,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        K1=0.01,
        K2=0.03,
        sigma=1.5)
    return ssim
