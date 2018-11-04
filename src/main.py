import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn

import os
import time
from time import sleep

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 50
learning_rate = 0.001
num_epochs = 20
counter = [0] * 7

class CSV2Dataset(data.Dataset):
    def __init__(self, path, training = True, validating = False, transforms = None):
        df = pd.read_csv(path)
        length = (int)(df.values.shape[0])
        if training:
            split_index = (int)(length * 0.85)
            if validating:
                split_data = df.values[split_index:]
            else:
                split_data = df.values[:split_index]
            features = [None] * split_data.shape[0]
            labels = [None] * split_data.shape[0]
            for i, row in enumerate(split_data):
                labels[i], feature = row
                counter[labels[i]] += 1
                features[i] = (np.array(feature.split(), dtype=float) / 255.).reshape([1, 48, 48])
            self.labels = labels
        else:
            features = [None] * length
            data_id = [None] * length
            for i, row in enumerate(df.values[:length]):
                data_id[i], features[i] = row
            self.data_id = data_id
        self.path = path
        self.training = training
        self.transforms = transforms
        self.features = features
        self.num = len(features)

    def __getitem__(self, index):
        if not self.transforms is None:
            features = self.transforms(self.features[index])
        if self.training:
            return self.features[index], self.labels[index]
        else:
            return self.features[index]

    def __len__(self):
        return self.num

data_dir = '../data/'
def get_data_loader(training = True, validating = False, shuffle = True):
    if training:
        transkey = 'train'
    else:
        transkey = 'test'
    dataset = CSV2Dataset(
        path = data_dir + transkey + '.csv',
        training = training,
        validating = validating,
    )
    dataloader = data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        # num_workers = 2
    )
    dataloader.num = dataset.num
    return dataloader

print('getting train data...')
train_data = get_data_loader()
print('getting validation data...')
validation_data = get_data_loader(validating = True)
all_data = {
    'train': train_data,
    'validation': validation_data
}
print(counter)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # (1, 48, 48)
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),      # (16, 48, 48)
            nn.ReLU(),                      # (16, 48, 48)
            nn.MaxPool2d(kernel_size = 2)   # (16, 24, 24)
        )
        self.conv2 = nn.Sequential(         # (16, 24, 24)
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),     # (32, 24, 24)
            nn.ReLU(),                      # (32, 24, 24)
            nn.MaxPool2d(kernel_size = 2)   # (32, 12, 12)
        )
        self.fc = nn.Linear(32 * 12 * 12, 7)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output

cnn = CNN().to(device)
# print(cnn)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr = learning_rate)
# print(train_data.num)
# print(validation_data.num)

best_model = cnn
best_acc = 0
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    for phase in ['train', 'validation']:
        running_data = all_data[phase]
        running_loss = 0.0
        running_acc = 0
        for i, (features, labels) in enumerate(running_data):
            features, labels = Variable(features).to(device), Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = cnn(features.float())
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_acc += torch.sum(predicted == labels)

        epoch_loss = running_loss / len(running_data)
        epoch_acc = running_acc.item() / running_data.num
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        if phase == 'validation' and epoch_acc > best_acc:
            best_acc = epoch_acc
            # torch.save(cnn.state_dict(), weight_file)
            best_model = cnn

print('Best validation accuracy: {:4f}'.format(best_acc))
# print(best_model.state_dict())
    



import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

cnn = vgg16(pretrained=True).cuda()
