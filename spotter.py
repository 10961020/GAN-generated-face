
import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from networks.resnet import resnet50
from networks.fake_spotter import fake_spotter
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dir', nargs='+', type=str, default=r'D:\zt\work\project\data\jpeg\eye_tell_all')
parser.add_argument('-m', '--model_path', type=str, default='weights/blur.pth')
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', type=int, default=4, help='number of workers')
parser.add_argument('-c', '--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--size_only', action='store_true', help='only look at sizes of images in dataset')
opt = parser.parse_args()

threshold = np.array([])
net_size = np.array([])
train_feature = np.array([])


def data_process():
    # Transform
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Dataset loader
    if type(opt.dir) == str:
        opt.dir = [opt.dir]

    print('Loading [%i] datasets' % len(opt.dir))
    data_loaders = []
    for dir in opt.dir:
        dataset = datasets.ImageFolder(dir, transform=trans)
        data_loaders += [torch.utils.data.DataLoader(dataset,
                                                     batch_size=opt.batch_size,
                                                     shuffle=False,
                                                     num_workers=opt.workers)]
    return data_loaders


def hook_sum(module, grad_input, grad_output):
    global threshold, net_size
    threshold = np.append(threshold, grad_output.cpu().numpy().sum())
    net_size = np.append(net_size, grad_output.cpu().numpy().size)


def hook_feature(module, grad_input, grad_output):
    global threshold, k, train_feature
    y = np.sum(grad_output.cpu().numpy() > threshold[k])
    k += 1
    train_feature = np.append(train_feature, y)


def model_load():
    # Load model
    model = resnet50(num_classes=1)
    if opt.model_path is not None:
        state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.eval()
    if not opt.use_cpu:
        model.cuda()
    return model


def feature_process():
    global threshold, k, train_feature, net_size

    # TODO 获取可训练层的平均值 记录为权重  第一步
    model = model_load()
    for name, module in model.named_modules():
        if "conv1" in name or "conv2" in name or "conv3" in name or 'downsample.0' in name or 'fc' in name:
            module.register_forward_hook(hook_sum)  # 求各个层的最大值或计算个数
    data_loaders = data_process()

    with torch.no_grad():
        for data_loader in data_loaders:
            for data, label in tqdm(data_loader):
                if not opt.use_cpu:
                    data = data.cuda()
                model(data).sigmoid().flatten().tolist()

    threshold = threshold.reshape(-1, 54)
    threshold = threshold.sum(axis=0)/(len(data_loader)*opt.batch_size*net_size[:54])

    # TODO 获取各个数据集的特征向量 记录为权重  第二步
    model = model_load()
    for name, module in model.named_modules():
        if "conv1" in name or "conv2" in name or "conv3" in name or 'downsample.0' in name or 'fc' in name:
            module.register_forward_hook(hook_feature)  # 求特征向量

    with torch.no_grad():
        for data_loader in data_loaders:
            for data, label in tqdm(data_loader):
                k = 0
                if not opt.use_cpu:
                    data = data.cuda()
                model(data).sigmoid().flatten().tolist()

    train_feature = train_feature.reshape(-1, 54)
    np.save('eye_tell_all_jpeg_feature.npy', train_feature)


def normalization(data):
    M_m = np.max(data, axis=1)-np.min(data, axis=1)
    return torch.tensor((data-np.min(data, axis=1).reshape(data.shape[0], -1)) / M_m.reshape(data.shape[0], -1)).type(torch.float32)


# fake 0
def train_fakespotter(config):
    dataset = np.load('train_feature.npy')
    label = [0 if i < config['data_size']/2 else 1 for i in range(config['data_size'])]
    np.random.seed(0)
    np.random.shuffle(dataset)
    np.random.seed(0)
    np.random.shuffle(label)
    train_dataset = dataset[:config['train_size'], :]
    train_label = label[:config['train_size']]
    test_dataset = dataset[config['train_size']:, :]
    test_label = label[config['train_size']:]

    model = fake_spotter()
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)

    for epoch in range(config['epoch']):

        model.train()
        for batch in range(0, config['train_size'], config['batch']):
            start_batch = time.time()
            if batch+config['batch'] < config['train_size']:
                x = train_dataset[batch:batch+config['batch'], :]
                label = train_label[batch:batch+config['batch']]
            else:
                x = train_dataset[batch:, :]
                label = train_label[batch:]

            x = normalization(x).to(device)
            label = torch.Tensor(label).unsqueeze(1).to(device)
            out = model(x)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(batch/config['train_size'], loss, time.time()-start_batch))

        print("---------------------------------------------------------------------\n验证：")
        model.eval()
        y_true, y_pred = [], []
        for batch in range(0, config['data_size']-config['train_size'], config['batch']):
            if batch+config['batch'] < config['data_size']-config['train_size']:
                x = test_dataset[batch:batch+config['batch'], :]
                y_true.extend(test_label[batch:batch+config['batch']])
            else:
                x = test_dataset[batch:, :]
                y_true.extend(test_label[batch:])

            x = normalization(x).to(device)
            y_pred.extend(model(x).sigmoid().flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = accuracy_score(y_true, y_pred > 0.5)
        print('Num reals: {}, Num fakes: {}'.format(np.sum(y_true), np.sum(1 - y_true)))
        print(acc)
        torch.save(model, config['output_path']+r'\{}-[accuracy_score]-{:.4f}.h5'.format(epoch, acc))


def val(config):
    dataset = np.load('eye_tell_all_jpeg_feature.npy')
    label = [0 if i < dataset.shape[0] / 2 else 1 for i in range(dataset.shape[0])]
    model = torch.load(os.path.join(config['output_path'],'49-[accuracy_score]-0.9895.h5'))
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model.eval()
    model.to(device)

    y_pred = []
    with torch.no_grad():
        for batch in range(0, dataset.shape[0], config['batch']):
            if batch + config['batch'] < dataset.shape[0]:
                x = dataset[batch:batch + config['batch'], :]
            else:
                x = dataset[batch:, :]

            x = normalization(x).to(device)
            y_pred.extend(model(x).sigmoid().flatten().tolist())

    y_true, y_pred = np.array(label), np.array(y_pred)
    print('Num reals: {}, Num fakes: {}'.format(np.sum(y_true), np.sum(1-y_true)))
    np.save('spotter_ture.npy', y_true)
    np.save('spotter_pred.npy', y_pred)

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
        'output_path': r'D:\zt\work\project\CNNDetection-master\checkpoints_fakespotter',
    }
    # feature_process()
    # train_fakespotter(config)
    val(config)


