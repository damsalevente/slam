'''
from https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py
'''
import torch
from torch import nn

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes # rx, ry, rz, tx, ty, tz

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 512)

        # for accel and gyro data
        self.sensor = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                )
        # RNN 
        self.cnnrnn = nn.LSTM(input_size = 256,
                            hidden_size = 512,
                            num_layers = 2,
                            batch_first = True)

        self.rnn_dropout1 = nn.Dropout(0.5)


        #final fully connected with the pose estimation
        self.fc2 = nn.Sequential(
                nn.Linear(1024+64, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_classes),
                nn.Tanh())

    def forward(self, x, sensordata):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out) # 1024 to 512

        out = out.view(x.size(0), 2, -1) # batch size, sequence len, fully connected output
        out, hc = self.cnnrnn(out)
        out = self.rnn_dropout1(out)
        #https://github.com/cezannec/capsule_net_pytorch/issues/4
        out = out.contiguous().view(x.size(0), -1)
        
        sens_out = self.sensor(sensordata) # 64 to 512
        #https://github.com/cezannec/capsule_net_pytorch/issues/4
        s_out = sens_out.contiguous().view(x.size(0), -1)
        
        concat = torch.cat((out, s_out), dim = 1)
        out = self.fc2(concat) # 512 to 6 

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)
