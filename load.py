import os
import cv2
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import GradientTape
from tensorflow.keras.losses import BinaryCrossentropy
from scipy.ndimage import binary_dilation, binary_erosion

from utils.process import get_crops_eye


def get_ids(dir_img):
    return (i for i in os.listdir(dir_img))


def split_train_val(dataset, batch_size, val_percent=0.1):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    if n % batch_size:
        n = n - n % batch_size
    random.shuffle(dataset)
    return {'train': dataset[:], 'val': dataset[-n:]}
    # return {'train': dataset[:-n], 'val': dataset[-n:]}


def get_imgs_and_masks(img_list, dir_img, face_detector, landmark_Predictor):
    """Return all the couples (img, y)"""
    imgs = to_cropped_imgs(img_list, dir_img, face_detector, landmark_Predictor)

    return imgs


def to_cropped_imgs(img_list, dir_img, face_detector, landmark_Predictor):
    ''' 图片处理 '''
    for id_img in img_list:
        input_file = os.path.join(dir_img, id_img)
        img = resize_and_crop(cv2.imread(input_file), input_file, face_detector, landmark_Predictor)
        y = 1 if 'real' in id_img else 0

        if isinstance(img, np.ndarray):
            yield img, y


def get_imgs(img_list, face_detector, landmark_Predictor):
    """Return all the couples (img, y)"""
    imgs = to_cropped_imgs1(img_list, face_detector, landmark_Predictor)

    return imgs


def to_cropped_imgs1(img_list, face_detector, landmark_Predictor):
    ''' 图片处理 '''
    for id_img in img_list:
        # input_file = os.path.join(dir_img, id_img)
        img = resize_and_crop(cv2.imread(id_img), id_img, face_detector, landmark_Predictor)
        y = 1 if 'real' in id_img else 0

        if isinstance(img, np.ndarray):
            yield img, y


def hwc_to_chw(img):
    return np.transpose(img, axes=[0, 3, 1, 2])


def normalize(x):
    return x / 255


def resize_and_crop(img, input_file, face_detector, landmark_Predictor):
    sclera_list = []
    img_eye, img_eye_mask = get_crops_eye(face_detector, landmark_Predictor, img, input_file)
    if not len(img_eye):
        return None

    for i in range(2):
        img_gray = cv2.cvtColor(img_eye[i], cv2.COLOR_BGR2GRAY)
        _, th2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)  # 前后景区分
        negative_mask = np.logical_not(img_eye_mask[i])
        negative_mask = binary_dilation(negative_mask)
        negative_mask = binary_dilation(negative_mask)
        th2[negative_mask] = 0

        # TODO 找到最大区域并填充
        contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area = []
        for j in range(len(contours)):
            area.append(cv2.contourArea(contours[j]))
        if area:
            max_idx = np.argmax(area)
            for k in range(len(contours)):
                if k != max_idx:
                    cv2.fillPoly(th2, [contours[k]], 0)

        # TODO 腐蚀
        try:
            th2 = binary_erosion(th2)
            th2 = binary_erosion(th2)

            mask_coords = np.where(th2 != 0)
            mask_min_y = np.min(mask_coords[0])
            mask_max_y = np.max(mask_coords[0])
            mask_min_x = np.min(mask_coords[1])
            mask_max_x = np.max(mask_coords[1])

            roi_top = np.clip(mask_min_y, 0, img_eye[i].shape[0])
            roi_bottom = np.clip(mask_max_y, 0, img_eye[i].shape[0])
            roit_left = np.clip(mask_min_x, 0, img_eye[i].shape[1])
            roi_right = np.clip(mask_max_x, 0, img_eye[i].shape[1])

            roi_image = img_eye[i][roi_top:roi_bottom, roit_left:roi_right, :]

            roi_image = cv2.resize(roi_image, (96, 96))
            sclera_list.append(roi_image)
        except:
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(img_eye[i][...,::-1])
            # plt.subplot(122)
            # plt.imshow(th2, cmap='gray')
            # plt.show()
            print(input_file)
            return None

    sclera_eye = np.concatenate([sclera_list[0],sclera_list[1]], axis=1)
    sclera_eye = cv2.resize(sclera_eye, (96, 96))

    return np.expand_dims(np.array(sclera_eye, dtype=np.float32), axis=0)


def batch(iterable, batch_size):
    """Yields lists by batch"""
    img = []
    y = []
    for i, t in enumerate(iterable):
        img.append(t[0])
        y.append(t[1])
        if (i + 1) % batch_size == 0:
            yield img, y
            img = []
            y = []

    if len(img) > 0:
        yield img, y


def list_transform_np(imgs):
    img_np = np.array([]).reshape((0, 96, 96, 3))

    for img in imgs:
        img_np = np.concatenate([img_np, img], axis=0)
    return img_np.astype(np.uint8)


def loss(model, img, y, training):
    loss_object = BinaryCrossentropy()
    y_ = model(img, training=training)
    return loss_object(y_true=y, y_pred=y_), y_


def grad(model, img, y):
    with GradientTape() as tape:
        loss_value, y_ = loss(model, img, y, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), y_
