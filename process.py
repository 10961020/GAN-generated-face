# import matlab.engine
import numpy as np
import dlib
import cv2
import os

from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.transform import hough_circle, hough_circle_peaks

import matplotlib.pyplot as plt
from skimage import filters
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# eng = matlab.engine.start_matlab()

LANDMARKS = {"mouth": (48, 68),
             "mouth_inner": (60, 68),
             "right_eyebrow":(17, 22),
             "left_eyebrow": (22, 27),
             "right_eye": (36, 42),
             "left_eye": (42, 48),
             "nose": (27, 35),
             "jaw": (0, 17),
             }

MOUTH_LM = np.arange(LANDMARKS["mouth_inner"][0], LANDMARKS["mouth"][1])
LEYE_LM = np.arange(LANDMARKS["left_eye"][0], LANDMARKS["left_eye"][1])
REYE_LM = np.arange(LANDMARKS["right_eye"][0], LANDMARKS["right_eye"][1])


def shape_to_np(shape):
    number_of_points = shape.num_parts
    points = np.zeros((number_of_points, 2), dtype=np.int32)
    for i in range(0, number_of_points):
        points[i] = (shape.part(i).x, shape.part(i).y)

    return points


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


# TODO 得到眼部区域的mask
def generate_convex_mask(shape, points_x, points_y):
    mask = np.zeros(shape, dtype=np.uint8)

    #clip to image size
    points_x = np.clip(points_x, 0, max(0, shape[1] - 1))
    points_y = np.clip(points_y, 0, max(0, shape[0] - 1))

    #set mask pixels
    mask[points_y, points_x] = 255
    mask = convex_hull_image(mask)

    return mask


# TODO 加载dlib人脸关键点检测
def load_facedetector(config):
    """Loads dlib face and landmark detector."""
    # download if missing http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    if not os.path.isfile(config['facedetector_path']):
        print('Could not find shape_predictor_68_face_landmarks.dat.')
        exit(-1)
    face_detector = dlib.get_frontal_face_detector()
    landmark_Predictor = dlib.shape_predictor(config['facedetector_path'])

    return face_detector, landmark_Predictor


# TODO 裁剪眼睛覆盖区域 320*280xp  返回裁剪后眼部区域以及对应的眼部mask
def get_crops_eye(face_detector, landmark_Predictor, img, input_file):
    faces = face_detector(img, 1)
    img_eye_crop = []
    img_eye_mask = []

    for face in faces:
        landmarks = landmark_Predictor(img, face)  # get 68 landmarks for each face
        landmarks_np = shape_to_np(landmarks)
        # for i, j in [(36, 39), (42, 45)]:
        for i in [LEYE_LM, REYE_LM]:
            eye_mark_local = landmarks_np[i]
            eye_mask = generate_convex_mask(img[..., 0].shape, eye_mark_local[..., 0], eye_mark_local[..., 1])
            eye_mask = eye_mask.astype('uint8')

            pt_pos_left, pt_pos_right = landmarks_np[i[0]], landmarks_np[i[3]]
            center_point = ((pt_pos_right[0] - pt_pos_left[0]) // 2 + pt_pos_left[0], (pt_pos_right[1] - pt_pos_left[1]) // 2 + pt_pos_left[1])
            try:
                img_eye_crop.append(img[center_point[1] - 70:center_point[1] + 70, center_point[0] - 80:center_point[0] + 80])
                img_eye_mask.append(eye_mask[center_point[1] - 70:center_point[1] + 70, center_point[0] - 80:center_point[0] + 80])
            except:
                print("123: ",input_file)

    return img_eye_crop, img_eye_mask


