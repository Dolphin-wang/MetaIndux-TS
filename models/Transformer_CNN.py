import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

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
        self.learning_rate = 1e-4                                       # 学习率
        self.num_engine = 0

        self.input_size = 14
        self.output_size = 1                                            # 类别数
        self.window_size = 50

        self.embedding = 32
        self.linear_hidden = 64
        self.num_head = 2
        self.num_encoder = 3

        self.cnn_size = 16
        
        print(self.device)
"""

效果比较好的，FD002 num_encoder = 3，num_head = 2, CNN_SZIE = 32,drop_out = 0  
"""

'''Attention Is All You Need'''

class Convlution(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Convlution, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,)
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
        self.fc2= nn.Linear(config.embedding * config.window_size, config.cnn_size ** 2)
        self.conv1 = Convlution(1,16)
        self.conv2 = Convlution(16,32)
        # self.conv3 = Convlution(60, 120)
        self.fc3 = nn.Sequential(
            nn.Linear(32 * config.cnn_size * config.cnn_size // 16 , config.linear_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_hidden, config.output_size),
        ) 


    def forward(self, x):
        # out = self.fc1(x) # out [128, 51, 32]
        # out = out.view(out.size(0),-1) # out [128, 51*32]
        out = self.fc2(x) # out [128, 32*32]
        out = out.view(out.size(0),-1,16).unsqueeze(1) # out [128,32,32]

        out = self.conv1(out) # out [128, 30, 16, 16]
        out = self.conv2(out) # out [128, 60, 8, 8]
        # out = self.conv3(out) # out [128, 120, 4, 4]
        out = out.view(out.size(0),-1)
        out = self.fc3(out)
        return out

class cnn_attention1(nn.Module):
    def __init__(self,config):
        super(cnn_attention1, self).__init__()
        length = config.window_size

        self.cnn_att = nn.Sequential(
            nn.Conv1d(config.embedding, 1,kernel_size=3,padding=1),
            nn.Linear(length,(length) // config.reduction),
            nn.ReLU(), 
            nn.Dropout(config.dropout),
            nn.Linear((length) // config.reduction, length),
            nn.Sigmoid()
        ) 

    def forward(self,out):
        att = self.cnn_att(out.permute(0,2,1))
        out = out * att.repeat(1, out.size(2) ,1).permute(0,2,1)  #out1[Bs,time,embed]
        return out



class TCNN1d(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, config):
        super(TCNN1d, self).__init__()
        self.window_size = config.window_size
        self.fc_embedding = nn.Linear(config.input_size , config.embedding)
        self.postion_embedding = Positional_Encoding(config.embedding, config.window_size , config.dropout, config.device)
        self.encoder = Encoder(config.embedding, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])
        self.attention = cnn_attention1(config)
        self.last_fc = nn.Sequential(
            nn.Linear(config.embedding * config.window_size , config.hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.output_size),
        ) 

    def forward(self, x):
        # x = x[:,50-self.window_size:,:]
        out1 = self.fc_embedding(x)
        out1 = self.postion_embedding(out1) # x(batch_size,window_size+1,input_size), out1(batch_size,window_size+1,input_size)
        for encoder in self.encoders:
            out1 = encoder(out1) #out1(64, 51, 32)
        # out1 = self.attention(out1)
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
