import torch


def blend_seg_pred(img, seg, alpha=0.5):
    pred = seg.argmax(1)
    pred = pred.view(pred.shape[0], 1, pred.shape[1], pred.shape[2]).repeat(1, 3, 1, 1)
    blend = img

    for i in range(1, seg.shape[1]):
        color_mask = -torch.ones_like(img)
        color_mask[:, -i, :, :] = 1
        alpha_mask = 1 - (pred == i).float() * alpha
        blend = blend * alpha_mask + color_mask * (1 - alpha_mask)

    return blend


