import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.mctan_utils import clones, LayerNorm, EncoderLayer, MultiHeadedAttention

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'MCTAN'
        self.save_path = 'weights/' + self.model_name + '.pth'        # 模型训练结果

        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

        self.dropout = 0.1                                              # 随机失活
        
        self.epoch = 100                                                 # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 15e-3                                       # 学习率
        self.num_engine = 0

        self.input_size = 14
        self.output_size = 1                                            # 类别数
        self.window_size = 50

        self.embedding = 32
        self.hidden = 64
        self.num_head = 2
        self.num_encoder = 4

        print(self.device)


class ChannelAttention(nn.Module):
    def __init__(self, maxlen, c=14, r=28, dropout=0.1):
        """
        calculate the channel attention
        :param maxlen: the input length
        :param c: input dim
        :param r: hidden dim, normally is set as 2 * input dim
        :param dropout:
        """
        super(ChannelAttention, self).__init__()
        self.w_0 = nn.Linear(maxlen, 1)
        self.w_1 = nn.Linear(c, r)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(r, c)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.weight = None

    def forward(self, x):
        x = x.transpose(1,2)
        weight = self.w_0(x).squeeze(2)
        weight = self.softmax(self.w_2(self.dropout(self.relu(self.w_1(weight))))).unsqueeze(2)
        self.weight = weight
        x = torch.mul(x, weight)
        return x.transpose(1, 2)

class Embeddings(nn.Module):
    "Time series linear projecting"

    def __init__(self, d_model, input_dim, maxlen, chanel_atten=False):
        """
        :param d_model: hidden dim
        :param input_dim: input dim
        :param maxlen: input length
        :param chanel_atten: True for applying channel attention
        """
        super(Embeddings, self).__init__()
        self.atten = ChannelAttention(maxlen, input_dim, 2*input_dim) if chanel_atten else None
        self.lut = nn.Linear(input_dim, d_model)

    def forward(self, x):
        if self.atten:
            x = self.atten(x)
            return self.lut(x)
        else:
            return self.lut(x)

class PositionalEncoding(nn.Module):
    "Position embedding."

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Multi_channel_temporal_attention_based_network(nn.Module):
    """
    A standard Encoder architecture for mctan
    """
    def __init__(self,output_dim, d_model, encoder, src_embed):
        super(Multi_channel_temporal_attention_based_network, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.fc1 = torch.nn.Linear(d_model, d_model//2)
        self.fc2 = torch.nn.Linear(d_model//2, output_dim)
        self.fc = torch.nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_mask=None):
        "Take in and process masked src and target sequences."
        return self.encode(src, src_mask)

    def encode(self, src, src_mask):
        out = self.encoder(self.src_embed(src), src_mask)
        return self.fc2(self.dropout(self.relu(self.fc1(out[:, -1]))))


def MCTAN(input_dim=1, output_dim=1, maxlen=50, stride=5, N=3, d_model=512, d_ff=2048, h=8, dropout=0.1,
                      fullattention=True, chanel_atten=False):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, fullattention, maxlen, stride, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, maxlen)
    model = Multi_channel_temporal_attention_based_network(output_dim,d_model,
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, input_dim, maxlen, chanel_atten), c(position)))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model