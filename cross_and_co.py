#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/29 13:39
# @Author  : zt
import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from networks.conet import conet, cross_conet
import matplotlib.pyplot as plt
from utils.load import get_imgs, batch_func
from sklearn.metrics import accuracy_score, roc_curve, auc


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


# TODO 数据预处理 3通道 6通道共生矩阵
def co_occurrences(config, cross_channel=False):
    cross = [(2, 0), (2, 1), (1, 0)]  # RG RB GB
    if not os.path.exists(config['output_path_eye_tell_all']):
        os.makedirs(config['output_path_eye_tell_all'])
    for root, dirs, files in os.walk(config['input_path']):
        for dir in dirs:
            if not os.path.exists(os.path.join(config['output_path_eye_tell_all'], dir)):
                os.makedirs(os.path.join(config['output_path_eye_tell_all'], dir))

        for file in files:
            img = hwc_to_chw(cv2.imread(os.path.join(root, file)))
            co_occurrences_matrix = np.empty((0, 256, 256))
            for i in range(3):
                channel_matrix = np.zeros((256, 256))
                img_gray1 = img[i, :, :-1].flatten()
                img_gray2 = img[i, :, 1:].flatten()
                matrix = list(zip(img_gray1, img_gray2))
                for j in range(len(matrix)):
                    channel_matrix[matrix[j]] += 1
                co_occurrences_matrix = np.concatenate((co_occurrences_matrix, np.expand_dims(channel_matrix, axis=0)))

            file_prefix, file_suffix = os.path.splitext(file)

            if cross_channel:
                for i in range(3):
                    j, k = cross[i]
                    channel_matrix = np.zeros((256, 256))
                    img_gray1 = img[j].flatten()
                    img_gray2 = img[k].flatten()
                    matrix = list(zip(img_gray1, img_gray2))
                    for j in range(len(matrix)):
                        channel_matrix[matrix[j]] += 1

                    co_occurrences_matrix = np.concatenate((co_occurrences_matrix, np.expand_dims(channel_matrix, axis=0)))
                np.save(os.path.join(config['output_path_eye_tell_all'], os.path.basename(root), file_prefix + '.npy'), co_occurrences_matrix)
            else:
                np.save(os.path.join(config['output_path_eye_tell_all'], os.path.basename(root), file_prefix + '.npy'), co_occurrences_matrix)


def normalization(data):
    data_t = data.reshape(data.shape[0], -1)
    M_m = np.max(data_t, axis=1)-np.min(data_t, axis=1)
    # print((data-np.min(data_t, axis=1).reshape((data.shape[0], 1))).shape)
    return torch.tensor((data-np.min(data_t, axis=1).reshape((data.shape[0], 1, 1, 1))) / M_m.reshape(data.shape[0], 1, 1, 1)).type(torch.float32)


def train_conet(config):
    model = conet()  # cross co
    # model = cross_conet()  # cross co
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print(device)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-6)

    # trans = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    for epoch in range(config['epoch']):
        model.train()
        train = get_imgs(config['output_path_co'])  # cross co
        val = get_imgs(config['output_path_co_val'])  # cross co

        for i, (x, y) in enumerate(batch_func(train, config['batch'])):
            start_batch = time.time()
            x = normalization(x).to(device)
            label = torch.Tensor(y).unsqueeze(1).to(device)
            out = model(x)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i*config['batch']/config['train_size'], loss, time.time()-start_batch))

        print("---------------------------------------------------------------------\n验证：")
        model.eval()
        y_true, y_pred = [], []
        for i, (x, y) in enumerate(batch_func(val, config['batch'])):
            x = normalization(x).to(device)
            y_pred.extend(model(x).sigmoid().flatten().tolist())
            y_true.extend(y)

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = accuracy_score(y_true, y_pred > 0.5)
        print('Num reals: {}, Num fakes: {}'.format(np.sum(y_true), np.sum(1 - y_true)))
        print(acc)
        torch.save(model, config['output_path_net']+r'\{}-[accuracy_score]-{:.4f}.h5'.format(epoch, acc))


def val(config):
    # model = torch.load(os.path.join(config['output_path_net'], '19-[accuracy_score]-0.9975.h5'))
    model = torch.load(os.path.join(config['output_path_net'], '48-[accuracy_score]-0.9715.h5'))
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model.eval()
    model.to(device)

    test = get_imgs(config['output_path_eye_tell_all'])
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(batch_func(test, config['batch'])):
            x = normalization(x).to(device)
            y_pred.extend(model(x).sigmoid().flatten().tolist())
            y_true.extend(y)

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print('Num reals: {}, Num fakes: {}'.format(np.sum(y_true), np.sum(1-y_true)))
    np.save('conet_ture.npy', y_true)
    np.save('conet_pred.npy', y_pred)

    # TODO ROC
    fp, tp, thr = roc_curve(y_true, y_pred)
    roc_auc = auc(fp, tp)
    plt.figure()
    lw = 2
    plt.plot(fp, tp, color='red', lw=lw, label='AUC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    config = {
        'epoch': 50,
        'batch': 16,
        'train_size': 10000,
        'data_size': 12000,
        'output_path_net': r'D:\zt\work\project\CNNDetection-master\checkpoints_conet',  # cross co
        'input_path': r'D:\zt\work\project\data\jpeg\eye_tell_all',
        'output_path_eye_tell_all': r'D:\zt\work\project\data\jpeg\eye_tell_all_co',
        'output_path_co': r'D:\zt\work\project\data\conet_data\train',
        'output_path_co_val': r'D:\zt\work\project\data\conet_data\val',
        'output_path_cross': r'D:\zt\work\project\data\cross_conet_data\train',
        'output_path_cross_val': r'D:\zt\work\project\data\cross_conet_data\val'
    }
    # co_occurrences(config)

    # train_conet(config)

    val(config)


