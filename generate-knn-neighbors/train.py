
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from Densenet import densenet121, densenet169, densenet201, densenet161
import os, shutil
from os.path import join

from tqdm import tqdm
import numpy as np
import random

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn


# exp settings
seed = 2020
exp_dir = 'res18_cars_seed{}'.format(seed)
net = resnet18(pretrained=True)
nb_class = 196
nb_epoch = 200
batch_size = 64
lr_begin = 0.01
use_apex = False
opt_level = "O1"
data_dir = '/home/xdjf/datasets/cars'
train_dir = join(data_dir, 'train')
val_dir = join(data_dir, 'val')
test_dir = join(data_dir, 'test')



# CUDA setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Random seed setting
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # multi gpu
torch.backends.cudnn.deterministic = False


# Model setting
for param in net.parameters():
    param.requires_grad = True

net.fc = nn.Linear(net.fc.in_features, nb_class)
# net.classifier = nn.Linear(net.classifier.in_features, nb_class)
net.cuda()


# Dataloader
train_transform = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

# val_transform = transforms.Compose([
#         transforms.Scale((550, 550)),
#         transforms.RandomCrop(448, padding=8),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
# val_set = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transform)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)
#
# test_transform = transforms.Compose([
#         transforms.Scale((550, 550)),
#         transforms.RandomCrop(448, padding=8),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
# test_set = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8)


# optimizer setting
CELoss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)


# Apex
if use_apex:
    from apex import amp
    net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)


# Training
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

shutil.copyfile('train.py', exp_dir + '/train.py')
shutil.copyfile('model.py', exp_dir + '/model.py')
shutil.copyfile('Resnet.py', exp_dir + '/Resnet.py')

min_train_loss = 9999

with open(os.path.join(exp_dir, 'train_log.csv'), 'w') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc, Val_Acc, Test_Acc\n')

impatient = 0

for epoch in range(nb_epoch):
    print('\n===== Epoch: {} ====='.format(epoch))
    lr_now = optimizer.param_groups[0]['lr']
    net.train()
    train_loss = train_correct = train_total = idx = val_correct = val_total = test_correct = test_total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, ncols=80)):
        idx = batch_idx
        if inputs.shape[0] < batch_size:
            continue
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        x, xc = net(inputs)
        loss = CELoss(xc, targets) * 1

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        _, predicted = torch.max(xc.data, 1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets.data).cpu().sum()
        train_loss += loss.item()

    scheduler.step()

    train_acc = 100. * float(train_correct) / train_total
    train_loss = train_loss / (idx + 1)
    print('Train | lr: {:.4f} | Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(lr_now, train_loss, train_acc, train_correct,  train_total))


    # # Evaluating
    # net.eval()
    #
    # with torch.no_grad():
    #
    #     for _, (inputs, targets) in enumerate(tqdm(val_loader)):
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #         x, xc = net(inputs)
    #
    #         _, predicted = torch.max(xc.data, 1)
    #
    #         val_total += targets.size(0)
    #         val_correct += predicted.eq(targets.data).cpu().sum()
    #
    # val_acc = 100. * float(val_correct) / val_total
    # print('Val | Acc: {:.3f}% ({}/{})'.format(val_acc, val_correct, val_total))
    #
    #
    # with torch.no_grad():
    #
    #     for _, (inputs, targets) in enumerate(tqdm(test_loader)):
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #         x, xc = net(inputs)
    #
    #         _, predicted = torch.max(xc.data, 1)
    #
    #         test_total += targets.size(0)
    #         test_correct += predicted.eq(targets.data).cpu().sum()
    #
    # test_acc = 100. * float(test_correct) / test_total
    # print('Test | Acc: {:.3f}% ({}/{})'.format(test_acc, test_correct, test_total))
    #
    #
    # # Logging
    # with open(os.path.join(exp_dir, 'train_log.csv'), 'a') as file:
    #     file.write('{}, {:.4f}, {:.3f}, {:.3f}%, {:.3f}%, {:.3f}%\n'
    #                .format(epoch, lr_now, train_loss, train_acc, val_acc, test_acc))
    #
    # # Save model
    # net.cpu()
    # # torch.save(net, os.path.join(exp_dir, 'epoch_{}.pth'.format(epoch)))
    # torch.save(net.state_dict(),
    #            os.path.join(exp_dir, 'epoch_{}_val_{:.2f}_test_{:.2f}.pth'.format(epoch, val_acc, test_acc)))
    # net.cuda()


    # Logging
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a') as file:
        file.write('{}, {:.4f}, {:.3f}, {:.3f}%\n'.format(epoch, lr_now, train_loss, train_acc))


    # Save model and stop training
    if train_loss < min_train_loss:
        impatient = 0
        min_train_loss = round(train_loss, 4)
        net.cpu()
        # torch.save(net, os.path.join(exp_dir, 'epoch_{}.pth'.format(epoch)))
        torch.save(net.state_dict(), os.path.join(exp_dir, 'min_loss.pth'))
        net.cuda()
    else:
        impatient += 1

    if impatient == 10:
        break