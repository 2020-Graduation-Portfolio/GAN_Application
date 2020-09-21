
import math
import numpy as np
import cv2
from fsgan.utils.bbox_utils import scale_bbox, crop_img


def _gaussian(size=3, sigma=0.25, amplitude=1, normalize=False, width=None, height=None, sigma_horz=None,
              sigma_vert=None, mean_horz=0.5, mean_vert=0.5):
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                    sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)

    return gauss


def draw_gaussian(image, point, sigma):
    point[0] = round(point[0], 2)
    point[1] = round(point[1], 2)

    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image

    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] = \
        image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1

    return image


def generate_heatmaps(height, width, points, sigma=None):

    sigma = max(1, int(np.round(width / 128.))) if sigma is None else sigma
    heatmaps = np.zeros((points.shape[0], height, width), dtype=np.float32)
    for i in range(points.shape[0]):
        if points[i, 0] > 0:
            heatmaps[i] = draw_gaussian(
                heatmaps[i], points[i], sigma)

    return heatmaps


def heatmap2rgb(heatmap):

    m = heatmap.mean(axis=0)
    rgb = np.stack((m, m, m), axis=-1)
    rgb *= 1.0 / rgb.max()

    return rgb


def hflip_face_landmarks(landmarks, width):

    landmarks = landmarks.copy()

    for p in landmarks:
        p[0] = width - p[0]

    right_jaw, left_jaw = list(range(0, 8)), list(range(16, 8, -1))
    landmarks[right_jaw + left_jaw] = landmarks[left_jaw + right_jaw]

    right_brow, left_brow = list(range(17, 22)), list(range(26, 21, -1))
    landmarks[right_brow + left_brow] = landmarks[left_brow + right_brow]

    right_nostril, left_nostril = list(range(31, 33)), list(range(35, 33, -1))
    landmarks[right_nostril + left_nostril] = landmarks[left_nostril + right_nostril]

    right_eye, left_eye = list(range(36, 42)), [45, 44, 43, 42, 47, 46]
    landmarks[right_eye + left_eye] = landmarks[left_eye + right_eye]

    mouth_out_right, mouth_out_left = [48, 49, 50, 59, 58], [54, 53, 52, 55, 56]
    landmarks[mouth_out_right + mouth_out_left] = landmarks[mouth_out_left + mouth_out_right]

    mouth_in_right, mouth_in_left = [60, 61, 67], [64, 63, 65]
    landmarks[mouth_in_right + mouth_in_left] = landmarks[mouth_in_left + mouth_in_right]

    return landmarks


def align_crop(img, landmarks, bbox, scale=2.0, square=True):
    right_eye_center = landmarks[36:42, :].mean(axis=0)
    left_eye_center = landmarks[42:48, :].mean(axis=0)

    eye_center = (right_eye_center + left_eye_center) / 2.0
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.)
    output = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    new_landmarks = np.concatenate((landmarks, np.ones((68, 1))), axis=1)
    new_landmarks = new_landmarks.dot(M.transpose())

    bbox_scaled = scale_bbox(bbox, scale, square)

    output, new_landmarks = crop_img(output, new_landmarks, bbox_scaled)

    return output, new_landmarks

