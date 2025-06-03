from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.save_path = 'weights/' + self.model_name + '.pth'        # 模型训练结果

        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

        self.dropout = 0.0                                              # 随机失活
        
        self.epoch = 100                                                 # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 2e-3                                     # 学习率
        self.num_engine = 0

        self.input_size = 14
        self.output_size = 1                                            # 类别数
        self.window_size = 21

        self.embedding = 32
        self.hidden = 128
        self.reduction = 8
        self.cnn_size = 32


'''

'''


class Convlution(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Convlution, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class CNN(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, config):
        super(CNN, self).__init__()
        self.fc1= nn.Linear(config.input_size, config.embedding)
        self.fc2= nn.Linear(config.embedding * config.window_size , config.cnn_size ** 2)
        self.conv1 = Convlution(1,16)
        self.conv2 = Convlution(16,32)#16,32
        # self.conv3 = Convlution(60, 120)
        self.fc3 = nn.Sequential(
            nn.Linear(32 * config.cnn_size * config.cnn_size // 16 , config.hidden ),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden , config.output_size),
        ) 


    def forward(self, x):
        out = self.fc1(x) # out [128, 51, 32]
        out = out.view(out.size(0),-1) # out [128, 51*32]
        out = self.fc2(out) # out [128, 32*32]
        out = out.view(out.size(0),-1,16).unsqueeze(1) # out [128,32,32]

        out = self.conv1(out) # out [128, 30, 16, 16]
        out = self.conv2(out) # out [128, 60, 8, 8]
        # out = self.conv3(out) # out [128, 120, 4, 4]
        out = out.view(out.size(0),-1)
        out = self.fc3(out)
        return out

class CNN1d(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, config):
        super(CNN1d, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(config.input_size, 10,kernel_size=9,padding=4),
            nn.ReLU(),
            nn.Conv1d(10, 10,kernel_size=9,padding=4),
            nn.ReLU(),
            nn.Conv1d(10, 10,kernel_size=9,padding=4),
            nn.ReLU(),
            nn.Conv1d(10, 10,kernel_size=9,padding=4),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.window_size * 10, 128),
            nn.ReLU(), 
            nn.Dropout(config.dropout),
            nn.Linear(128, config.output_size),

        ) 

    def forward(self, x):
        out = self.conv1d( x.permute(0,2,1) )
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
