from heapq import nsmallest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch.nn.modules import conv
from models.Transformer import Encoder,Positional_Encoding,Position_wise_Feed_Forward
from models.Transformer_CNN import CNN

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TEST'
        self.save_path = 'weights/' + self.model_name + '.pth'        # 模型训练结果

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

        self.dropout = 0.1                                              # 随机失活
        
        self.epoch = 100                                                 # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.learning_rate = 3e-4                                   # 学习率
        self.num_engine = 0

        self.input_size = 14
        self.output_size = 1                                            # 类别数
        self.window_size = 50


        ###lstm
        self.lstm_hidden = 32 #16
        self.num_layers = 1

        ### transfomer
        self.embedding = 32 #32
        self.linear_hidden = 64 #64
        self.num_head = 1
        self.num_encoder = 1

        self.cnn_size = 16
        print(self.device)
"""
num——head = 1
num——encoder = 1
dropout = 0.0
"""
class TLSTM(nn.Module):
    def __init__(self,config) -> None:
        super(TLSTM,self).__init__()
        self.fc_embedding = nn.Sequential(
                nn.Linear(config.input_size , 48),
                nn.ReLU(),
                nn.Linear(48, config.embedding)
        )

        self.encoder = Encoder(config.embedding, config.num_head, config.linear_hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder) 
            for _ in range(config.num_encoder)])
        self.encoder2 = Encoder(config.embedding, config.num_head, config.linear_hidden, config.dropout)

        self.lstm1 = nn.LSTM(config.embedding, config.lstm_hidden, config.num_layers, bidirectional=False,
                        batch_first=True, dropout=config.dropout)#batch_first代表输入数据的第一个维度是batch_size

        self.last_fc = nn.Sequential(
            nn.Linear(config.embedding * config.window_size , config.output_size),
        ) 
        self.convs = CNN(config)
        self.layer_norm = nn.LayerNorm(config.embedding)
    def forward(self, x):
        out1 = self.fc_embedding(x)
        out2, _ = self.lstm1(out1)
        for encoder in self.encoders:
            out1 = encoder(out1) #out1(64, 51, 32)
        out3 = self.layer_norm(out2 + out1)
        out4 = self.encoder2(out3) #out1(64, 51, 32)
        # out5 = out4.reshape(out4.size(0), -1)
        # out6 = self.last_fc(out5)  # 句子最后时刻的 hidden state
        out5 = out4.view(out4.size(0),-1) #cnn
        out6 = self.convs(out5)

        return out6

class TFS(nn.Module):
    def __init__(self,config) -> None:
        super(TFS,self).__init__()
        self.fc_embedding = nn.Sequential(
                nn.Linear(config.input_size , 48),
                nn.ReLU(),
                nn.Linear(48, config.embedding)
        )
        
        self.encoder = Encoder(config.embedding, config.num_head, config.linear_hidden, config.dropout)
        self.feature_weight = nn.Sequential(
                nn.Linear(config.window_size , 1),
                # nn.Tanh(),
                # nn.Linear(25 , 1),
        )
        self.encoder2 = Encoder(config.embedding, config.num_head, config.linear_hidden, config.dropout)

        self.lstm1 = nn.LSTM(config.embedding, config.lstm_hidden, config.num_layers, bidirectional=False,
                        batch_first=True, dropout=config.dropout)#batch_first代表输入数据的第一个维度是batch_size

        self.last_fc = nn.Sequential(
            nn.Linear(config.embedding * config.window_size , config.linear_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_hidden, config.output_size), 
        ) 
        self.convs = CNN(config)
        self.layer_norm = nn.LayerNorm(config.embedding)
    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        out1 = self.fc_embedding(x)
        # out1 = x
        out2, _ = self.lstm1(out1)
        
        fw1 = self.feature_weight(out1.transpose(1,2))
        fw2 = F.softmax(fw1,dim=1)
        _, idx1 = torch.sort(fw2.squeeze(-1), descending=False)
        # print(idx1[:,:3])
        _ = torch.ones_like(fw2.squeeze(-1))
        for i in range(len(_)):
            _[i,idx1[i,:10]]=0
        A = (fw2.squeeze(-1)*_).unsqueeze(-1)
        fw3 = torch.mul(out1.transpose(0,1), A.squeeze(-1))
        # 上面是attention机制

        out1 = fw3.transpose(0,1)

        out3 = self.layer_norm( out2 + out1)
        out4 = self.encoder2(out3) #out1(64, 51, 32)
        # out4 = out3
        # out5 = out4.reshape(out4.size(0), -1)
        # out6 = self.last_fc(out5)  # 句子最后时刻的 hidden state
        out5 = out4.view(out4.size(0),-1) #cnn
        out6 = self.convs(out5)

        return out6

if __name__ == "__main__":
    config = Config()
    model = TFS(config).to(config.device)
    input = torch.rand(8,50,14).cuda()
    output = model(input)
    print(output.shape)
