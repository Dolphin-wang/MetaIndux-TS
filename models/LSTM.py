from unicodedata import bidirectional
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import dropout


"""
rmse 12.6
"""
class Config(object):
    def __init__(self):
        self.model_name = 'LSTM'
        self.save_path = 'weights/' + self.model_name + '.pth'        # 模型训练结果

        self.epoch = 100
        self.batch_size = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
        

        self.dropout = 0.1
        self.learning_rate = 2e-3

        self.input_size = 14
        self.output_size = 1
        self.window_size = 48

        self.lstm_hidden = 128
        self.num_layers = 1
        self.embedding = 48
        self.hidden = 128
        
class LSTM(nn.Module):
    def __init__(self,config) -> None:
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(config.input_size, config.input_size*2, config.num_layers,
                        batch_first=True)#batch_first代表输入数据的第一个维度是batch_size
        self.last_fc = nn.Sequential(
            nn.Linear(config.input_size*2 * config.window_size , config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.output_size),
        ) 

    def forward(self, x):
        out2, _ = self.lstm(x) #out1 [batch_size,time_step,hidden_size] 
        out3 = out2.reshape(out2.size(0), -1)
        out4 = self.last_fc(out3)  # 句子最后时刻的 hidden state

        return out4

class LSTMClassifier(nn.Module):
    def __init__(self,config) -> None:
        super(LSTMClassifier,self).__init__()
        self.lstm = nn.LSTM(config.input_size, config.input_size // 2, config.num_layers,
                        batch_first=True)#batch_first代表输入数据的第一个维度是batch_size
        self.last_fc = nn.Linear(config.input_size // 2, config.output_size)
        self.sigmoid = nn.Sigmoid()  # 添加sigmoid层
    def forward(self, x):
        out2, _ = self.lstm(x) #out1 [batch_size,time_step,hidden_size] 
        out3 = self.last_fc(out2[:,-1,:])  # 句子最后时刻的 hidden state
        output = self.sigmoid(out3)
        return output

