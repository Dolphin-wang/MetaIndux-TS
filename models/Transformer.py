import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch.nn.modules import transformer

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.save_path = 'weights/' + self.model_name + '.pth'        # 模型训练结果

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

        self.dropout = 0.00                                              # 随机失活
        
        self.epoch = 100                                                 # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 2e-5                                       # 学习率
        self.num_engine = 0

        self.input_size = 14
        self.output_size = 1                                            # 类别数
        self.window_size = 48

        self.embedding = 32
        self.hidden = 128
        self.num_head = 1
        self.num_encoder = 3

        print(self.device)
"""

效果比较好的，FD002 num_encoder = 5，num_head = 2, dropout = 0.0
"""

'''Attention Is All You Need'''

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.window_size = config.window_size
        self.fc_embedding = nn.Linear(config.input_size , config.embedding)
        self.postion_embedding = Positional_Encoding(config.embedding, config.window_size , config.dropout, config.device)
        self.encoder = Encoder(config.embedding, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        self.last_fc = nn.Sequential(
            nn.Linear(config.embedding * config.window_size , config.hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.output_size),
        ) 

    def forward(self, x):
        out1 = self.fc_embedding(x)
        out1 = self.postion_embedding(out1) # x(batch_size,window_size+1,input_size), out1(batch_size,window_size+1,input_size)
        for encoder in self.encoders:
            out1 = encoder(out1) #out1(64, 51, 32)
        out2 = out1.view(out1.size(0), -1) #shape(64, 51 * 32)
        out3 = self.last_fc(out2) #out3(64, 1)
        return out3

class transformer_torch(nn.Module):
    def __init__(self, config):
        super(transformer_torch, self).__init__()
        self.fc_embedding = nn.Linear(config.input_size, config.embedding)
        self.postion_embedding = Positional_Encoding(config.embedding, config.window_size, config.dropout, config.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.embedding,nhead=1)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=3)
        self.last_fc = nn.Sequential(
            nn.Linear(config.embedding * config.window_size , config.hidden ),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.output_size),
        ) 
    def forward(self, x):
        out1 = self.fc_embedding(x)
        out1 = self.postion_embedding(out1) # x(batch_size,window_size+1,input_size), out1(batch_size,window_size+1,input_size)
        out1 = self.encoder(out1) #out1(64, 51, 32)
        out2 = out1.view(out1.size(0), -1) #shape(64, 51 * 32)
        out3 = self.last_fc(out2) #out3(64, 1)
        return out3

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out1 = self.attention(x)
        out2 = self.feed_forward(out1)
        return out2


class Positional_Encoding(nn.Module):
    def __init__(self, embed, seq_len, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(seq_len)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1)) #matmul,矩阵乘法   permute用于维度转换，原本为（0，1，2）进行换位即可
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head) #这相当于一个方阵
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)#相当于reshape
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
