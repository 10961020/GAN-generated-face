#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time     :2022/12/7 14:51
# @Author   :tong.z

import cv2
import os
import time
import glob
import dlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers, metrics, models
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import binary_dilation, binary_erosion

from utils.model_utils import ResidualAttentionNetwork
from utils.process import load_facedetector
from utils.load import get_ids, split_train_val, get_imgs_and_masks, batch, grad, list_transform_np, normalize, get_imgs


config = {
    'input_path': r'E:\zhangtong\bloodshot_research\face_img\train',
    'input_path_val': r'E:\zhangtong\bloodshot_research\face_img\val',
    'output_path': r'E:\zhangtong\bloodshot_research\output',
    'facedetector_path': r'E:\zhangtong\eye_tell_all/shape_predictor_68_face_landmarks.dat',
}


def train_net(net, epochs=100, batch_size=4, lr=0.001, val_percent=0.1, save_cp=True, gpu=True,):

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    face_list = get_ids(config['input_path'])
    face_dataset = split_train_val(face_list, batch_size, val_percent)
    face_detector, landmark_Predictor = load_facedetector(config)

    train_loss_results = []
    train_accuracy_results = []

    N_train = len(face_dataset['train'])

    lr_list = np.linspace(lr, 1e-5, epochs)

    for epoch in range(epochs):
        # net.train()
        start_epoch = time.time()

        lr = lr_list[epoch]
        optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9)

        epoch_loss_avg = metrics.Mean()
        epoch_accuracy = metrics.Accuracy()

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        train = get_imgs_and_masks(face_dataset['train'], config['input_path'], face_detector, landmark_Predictor)
        # val = get_imgs_and_masks(face_dataset['val'], config['input_path_real'], face_detector, landmark_Predictor)

        for i, (x, y) in enumerate(batch(train, batch_size)):
            start_batch = time.time()

            x = list_transform_np(x)
            x = normalize(x)
            y = np.array(y).reshape(-1,1)

            loss_value, grads, y_ = grad(net, x, y)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))


            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, net(x, training=True))

            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * batch_size / N_train, loss_value, time.time()-start_batch))


        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

        net.save(config['output_path'] + '\{}-[train_loss]-{:.4f}.h5'.format(epoch, epoch_loss_avg.result() / epoch))

        print('Spend time: {:.3f}s'.format(time.time() - start_epoch))
    print()


def val(batch_size=1):
    net = models.load_model(os.path.join(config['output_path'], '15-[train_loss]-0.0000.h5'))
    real_img = glob.glob(os.path.join(config['input_path_val'], 'real_*.png'))
    style_img = glob.glob(os.path.join(config['input_path_val'], 'style_*.png'))

    face_detector, landmark_Predictor = load_facedetector(config)

    y_real = []
    y_style = []


    real_next = get_imgs(real_img, face_detector, landmark_Predictor)
    style_next = get_imgs(style_img, face_detector, landmark_Predictor)


    for i, (x, y) in enumerate(batch(real_next, batch_size)):
        x = list_transform_np(x)
        x = normalize(x)
        y_ = net(x)

        y_real.append(np.array(y_)[0][0])
    # y_real = np.array(y_real)
    # print(np.sum(y_real>0.5))

    for i, (x, y) in enumerate(batch(style_next, batch_size)):
        x = list_transform_np(x)
        x = normalize(x)
        y_ = net(x)
        y_real.append(np.array(y_)[0][0])

    y_real = np.array(y_real).reshape(-1)
    y = np.array([1 if i < 96 else 0 for i in range(194)]).reshape(-1)
    fp, tp, thr = roc_curve(y, y_real)
    roc_auc = auc(fp, tp)
    plt.figure()
    lw = 2
    plt.plot(fp, tp, color='red', lw=lw, label='AUC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    print()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # model = ResidualAttentionNetwork((96, 96, 3), 1).attention_mod()
    model = models.load_model(os.path.join(config['output_path'], '8-[train_loss]-0.0894.h5'))
    train_net(model)
    # val(model)



