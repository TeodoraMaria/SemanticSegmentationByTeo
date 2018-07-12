import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Helpers.ColorMapLabels import color_map
from Helpers.ImageSizeDecider import my_transform_image
from MyDataSet import MyDataSet
from model import TeoNet
import torch.nn.functional as F
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main():
    color_map()

    transform_train = transforms.Compose([
        transforms.Resize(256)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    current_directory = os.getcwd()
    train_set = MyDataSet(filename='trainval.txt', img_dir=os.path.join(current_directory, r'train_images'),
                          transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2,
                                               shuffle=True, num_workers=0)

    test_set = MyDataSet(filename='test.txt',img_dir=os.path.join(current_directory, r'test_images'),
                         transform=transform_test)

    validation_loader = torch.utils.data.DataLoader(test_set, batch_size=2,
                                                    shuffle=False,
                                                    num_workers=0)

    if torch.cuda.is_available():
        net = TeoNet().cuda()

    if torch.cuda.device_count() > 1:
        device_ids = range(torch.cuda.device_count())
        net = nn.DataParallel(net, device_ids=device_ids)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 300
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        total = 0
        correct = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # loss = Variable(loss, requires_grad=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total = targets.size(0)*targets.size(1)*targets.size(2)
            # outputs = F.softmax(outputs, 1)
            _, out = torch.max(outputs, 1)
            correct = torch.sum(torch.eq(out, targets))
        print('Results after epoch %d' % (epoch + 1))

        print('Training Loss: %.3f | Training accuracy: %.3f%%'
              % (train_loss / (batch_idx + 1),
              100. *correct / total))

        validation(net, validation_loader, criterion)

    torch.save(net.state_dict(), "TeoNetOverfit")
    im = transform_test(my_transform_image(500, 500,
                                           "{}/{}".format(os.path.join(current_directory, r'train_images'),
                                           '2007_000063.jpg')))
    im = im * 255
    # im = im.view(1,im.size(0),im.size(1),im.size(2))
    im.unsqueeze_(0)
    if torch.cuda.is_available():
        im = im.cuda()
    im = Variable(im)
    out = net(im)
    # out = F.softmax(out, 1)
    _, out = torch.max(out, 1)
    out = out.view(out.size(1), out.size(2))
    plt.imshow(out.cpu().numpy())
    plt.show()


def validation(net, validation_loader, criterion):
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(validation_loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)

        loss = criterion(outputs, targets)

        valid_loss += loss.item()
        total += targets.size(0)
    print('Validation Loss: %.3f'
          % (valid_loss / (batch_idx + 1)))

if __name__ == '__main__':
    main()