import os
import face_alignment
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from glob import glob
import torchvision.transforms.functional as F
from fsgan.utils.bbox_utils import get_main_bbox
from fsgan.utils.img_utils import rgb2tensor
from fsgan.utils.bbox_utils import scale_bbox, crop_img


def extract_landmarks_bboxes_euler_from_images(img_dir, face_pose, face_align=None, img_size=(224, 224),
                                              scale=1.2, device=None, cache_file=None):

    if face_align is None:
        face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

    cache_file = img_dir + '.pkl' if cache_file is None else cache_file
    if not os.path.exists(cache_file):
        frame_indices = []
        landmarks = []
        bboxes = []
        eulers = []

        img_paths = glob(os.path.join(img_dir, '*.jpg'))

        for i, img_path in tqdm(enumerate(img_paths), unit='images', total=len(img_paths)):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = img_bgr[:, :, ::-1]
            detected_faces = face_align.face_detector.detect_from_image(img_bgr.copy())

            if len(detected_faces) == 0:
                continue
            curr_bbox = get_main_bbox(np.array(detected_faces)[:, :4], img_bgr.shape[:2])
            detected_faces = [curr_bbox]

            preds = face_align.get_landmarks(img_rgb, detected_faces)
            curr_landmarks = preds[0]

            curr_bbox[2:] = curr_bbox[2:] - curr_bbox[:2] + 1

            scaled_bbox = scale_bbox(curr_bbox, scale)
            cropped_frame_rgb, cropped_landmarks = crop_img(img_rgb, curr_landmarks, scaled_bbox)
            scaled_frame_rgb = np.array(F.resize(Image.fromarray(cropped_frame_rgb), img_size, Image.BICUBIC))
            scaled_frame_tensor = rgb2tensor(scaled_frame_rgb.copy()).to(device)
            curr_euler = face_pose(scaled_frame_tensor)
            curr_euler = np.array([x.cpu().numpy() for x in curr_euler])

            frame_indices.append(i)
            landmarks.append(curr_landmarks)
            bboxes.append(curr_bbox)
            eulers.append(curr_euler)

        frame_indices = np.array(frame_indices)
        landmarks = np.array(landmarks)
        bboxes = np.array(bboxes)
        eulers = np.array(eulers)

        with open(cache_file, "wb") as fp:
            pickle.dump(frame_indices, fp)
            pickle.dump(landmarks, fp)
            pickle.dump(bboxes, fp)
            pickle.dump(eulers, fp)
    else:
        with open(cache_file, "rb") as fp:
            frame_indices = pickle.load(fp)
            landmarks = pickle.load(fp)
            bboxes = pickle.load(fp)
            eulers = pickle.load(fp)

    return frame_indices, landmarks, bboxes, eulers
