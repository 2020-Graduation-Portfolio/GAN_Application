
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.utils as vutils


def rgb2tensor(img, normalize=True):
    if isinstance(img, (list, tuple)):
        return [rgb2tensor(o) for o in img]
    tensor = F.to_tensor(img)
    if normalize:
        tensor = F.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return tensor.unsqueeze(0)


def bgr2tensor(img, normalize=True):
    if isinstance(img, (list, tuple)):
        return [bgr2tensor(o, normalize) for o in img]
    return rgb2tensor(img[:, :, ::-1].copy(), normalize)


def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2rgb(img_tensor):
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img * 255).astype('uint8')

    return output_img


def tensor2bgr(img_tensor):
    output_img = tensor2rgb(img_tensor)
    output_img = output_img[:, :, ::-1]

    return output_img


def make_grid(*args, cols=8):
    if len(args) == 0:
        raise RuntimeError('At least one input tensor must be given!')
    step = args[0].shape[0]
    cols = min(step, cols)
    imgs = []
    for d in range(0, args[0].shape[0], cols):
        for arg in args:
            for i in range(d, min(d + cols, step)):
                imgs.append(arg[i])

    return vutils.make_grid(imgs, nrow=cols, normalize=True, scale_each=False)


def create_pyramid(img, n=1):
    if isinstance(img, (list, tuple)):
        return img

    pyd = [img]
    for i in range(n - 1):
        pyd.append(nn.functional.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False))

    return pyd
