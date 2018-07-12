import torch.nn as nn
import math
import torch.nn.functional as F
import torch


class TeoNet(nn.Module):
    def __init__(self):
        super(TeoNet, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.fc6 = nn.Conv2d(512,64, kernel_size=1)
        # self.fc6 = nn.Linear(512*8*8, 4096)
        self.fc7 = nn.Conv2d(64,64, kernel_size=1)
        # self.fc7 = nn.Linear(4096, 4096)

        self.fc8 = nn.Conv2d(64, 21, kernel_size=1)
        self.layer9 = nn.ConvTranspose2d(21, 512, kernel_size=4, stride=2, padding= 1)
        self.layer10 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding= 1)
        self.layer11 = nn.ConvTranspose2d(512, 21, kernel_size=16, stride=8, padding= 4)

    def forward(self, x):
        out1 = F.relu(self.conv1_1(x))
        out1 = F.relu(self.conv1_2(out1))
        out1_2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.conv2_1(out1_2))
        out2 = F.relu(self.conv2_2(out2))
        out2_3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.conv3_1(out2_3))
        out3 = F.relu(self.conv3_2(out3))
        out3 = F.relu(self.conv3_3(out3))
        out3 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.conv4_1(out3))
        out4 = F.relu(self.conv4_2(out4))
        out4 = F.relu(self.conv4_3(out4))
        out4 = F.max_pool2d(out4, kernel_size=2, stride=2)
        out5 = F.relu(self.conv5_1(out4))
        out5 = F.relu(self.conv5_2(out5))
        out5 = F.relu(self.conv5_3(out5))
        out5_6 = F.max_pool2d(out5, kernel_size=2, stride=2)

        # out = out5_6.view(out5_6.size(0), -1)

        out = F.relu(self.fc6(out5_6))
        # out = F.dropout2d(out)
        out = F.relu(self.fc7(out))
        # out = F.dropout2d(out)

        # out = out.view(out.size(0), -1, 8, 8)

        out8 = F.relu(self.fc8(out))
        out9 = F.relu(self.layer9(out8))
        out = torch.cat((out9,out4),1)
        out10 = F.relu(self.layer10(out))
        out = torch.cat((out10,out3), 1)
        # out11 = F.softmax(self.layer11(out), dim=1)
        out = self.layer11(out)

        # _, out = torch.max(out11, 1)

        # out = out.view(out.size(0),1,out.size(1),out.size(2))

        return out
