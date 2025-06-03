import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'DNN'
        self.save_path = 'weights/' + self.model_name + '.pth'        # 模型训练结果

        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

        self.dropout = 0.0                                              # 随机失活
        
        self.epoch = 100                                                 # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 7e-4                                       # 学习率
        self.num_engine = 0

        self.input_size = 14
        self.output_size = 1                                            # 类别数
        self.window_size = 40

        self.embedding = 14
        self.linear_hidden = 128
        self.hidden = [200,200,32]



'''Convolutional Neural Networks for Sentence Classification'''


class DNN(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, config):
        super(DNN, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(config.input_size, config.hidden[0]),
            nn.ReLU(),
            nn.Linear(config.hidden[0], config.hidden[1]),
            nn.ReLU(),
            nn.Linear(config.hidden[1], config.hidden[2]),
            nn.ReLU(),
        ) 

        self.last_fc = nn.Sequential(
            nn.Linear(config.window_size * config.hidden[2], config.linear_hidden),
            nn.ReLU(),
            nn.Linear(config.linear_hidden, config.output_size),
        ) 
    def forward(self, x):
        out = self.fc1(x)
        out = out.view(out.size(0),-1) # out [128, 51*32]
        out = self.last_fc(out) # out [128, 1]
        return out