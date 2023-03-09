
import shutil
import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from utils.process import *
import utils.frangi as frangi

config = {
    'input_path': r'E:\zhangtong\data\bloodshot',
    'output_path': r'E:\zhangtong\bloodshot_research\output',
    "save_img": r'E:\zhangtong\bloodshot_research\realImg',
    'facedetector_path': r'E:\zhangtong\eye_tell_all/shape_predictor_68_face_landmarks.dat',
}


def main():
    file_list = []
    bIou_max = [[],[]]  # style 0 ffhq 1
    cla_sum = []
    cla_max = []
    num = 0

    if os.path.isdir(config['input_path']):
        file_list = [os.path.join(config['input_path'], name) for name in os.listdir(config['input_path'])]

    if len(file_list) == 0:
        print('No files at given input path.')
        exit(-1)

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
    if not os.path.exists(config['save_img']):
        os.makedirs(config['save_img'])

    face_detector, landmark_Predictor = load_facedetector(config)

    for input_file in tqdm(file_list):
        print(input_file)
        img = cv2.imread(input_file)

        if img is None or img is False:
            print("Could not open image file: %s" % input_file)
            continue

        img_eye, img_eye_mask = get_crops_eye(face_detector, landmark_Predictor, img, input_file)

        if not len(img_eye):
            continue

        # # TODO 保存眼部图片到文件
        # i, j = os.path.splitext(input_file.split('\\')[-1])
        # cv2.imwrite(os.path.join(config['output_path'], cla[value] + '_' + i + '_l' + j), img_eye[0])
        # cv2.imwrite(os.path.join(config['output_path'], cla[value] + '_' + i + '_r' + j), img_eye[1])
        # continue

        value = 0
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
            th2 = binary_erosion(th2)
            th2 = binary_erosion(th2)
            th2negative_mask = np.logical_not(th2)

            blood = cv2.normalize(img_gray.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            outIm = frangi.FrangiFilter2D(blood)
            img_ = outIm * 10000
            img_[th2negative_mask] = 0
            img_ = np.where(img_ > 0.004, 1, 0)
            img_ = img_.astype(np.uint8)

            if np.sum(img_)>0:
                value = 1
                break
        if value:
            i, j = os.path.splitext(input_file.split('\\')[-1])
            cv2.imwrite(os.path.join(config['save_img'], i + j), img)



if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    main()
    # for roots, dirs, files in os.walk(r'E:\zhangtong\bloodshot_research\face_img\STYLEGAN_30'):
    #     for file in files:
    #         i, j = os.path.splitext(file.split('\\')[-1])
    #         os.replace(os.path.join(roots, file), os.path.join(roots, 'style_'+i+j))