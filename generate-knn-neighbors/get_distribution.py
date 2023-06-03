
import os, shutil

from tqdm import tqdm
import pandas as pd

import torch
from torchvision import transforms
import torch.nn as nn
from Folder import ImageFolder

from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from Densenet import densenet121, densenet169, densenet201, densenet161

from os.path import join


# settings
# exp_dir = 'r18_cars_test_seed2020'
exp_dir = 'G:\DNN提取规则\example'
net = resnet18(pretrained=True)
# img_dir = '/home/gpc/disk_1/datasets/cars'
img_dir = r'G:\DNN提取规则\example\r18_seed2020_x4_distance_to_center_all.csv'
nb_class = 196
batch_size = 4
model_path = os.path.join(exp_dir, 'min_loss.pth')
attr_name = exp_dir


# CUDA setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Transforms
transform_test = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# Model setting
net.fc = nn.Linear(net.fc.in_features, nb_class)  # for resnet
# net.classifier = nn.Linear(net.classifier.in_features, nb_class)  # for densenet
# net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load(model_path))

net.cuda()
net.eval()


# Testing
shutil.copyfile('get_distribution.py', exp_dir + '/get_distribution.py')

for data_set in ['train', 'val', 'test']:
# for data_set in ['test']:
    testset = ImageFolder(root=os.path.join(img_dir, data_set),
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)

    distribution_x = []
    distribution_xc = []
    predictions = []
    labels = []
    paths = []
    test_loss = correct = total = 0

    with torch.no_grad():

        for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):
            inputs, targets = inputs.cuda(), targets.cuda()
            # x1, x2, x3, x4, xc = net(inputs)
            x, xc = net(inputs)

            _, predicted = torch.max(xc.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            distribution_x.extend(x.cpu().numpy())
            distribution_xc.extend(xc.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            labels.extend(targets.tolist())
            paths.extend(path)


    test_acc = 100. * float(correct) / total
    print('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

    pd.DataFrame(labels).to_csv(join(exp_dir, 'labels_{}.csv'.format(data_set)), header=None, index=None)
    pd.DataFrame(predictions).to_csv(join(exp_dir, 'predictions_{}.csv'.format(data_set)), header=None, index=None)
    pd.DataFrame(paths).to_csv(join(exp_dir, 'paths_{}.csv'.format(data_set)), header=None, index=None)

    pd.DataFrame(distribution_x).to_csv(join(exp_dir, 'distribution_x_{}.csv'.format(data_set)), header=None, index=None)
    pd.DataFrame(distribution_xc).to_csv(join(exp_dir, 'distribution_xc_{}.csv'.format(data_set)), header=None, index=None)
    #
    # print('csvs saved.\n'.format(data_set))