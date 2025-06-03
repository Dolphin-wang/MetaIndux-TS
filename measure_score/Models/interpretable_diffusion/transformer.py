import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp


class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3#设置多项式拟合的阶数
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )#使用序列容器构建趋势模型，包括卷积层，维度转置，将通道维度放在最前面，最后再一个卷积层

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)#创建用于多项式拟合的线性空间
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)#构建多项式空间，用于拟合趋势

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)#对输入数据进行趋势建模，然后进行维度转置以匹配多项式空间的维度
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))#计算趋势值，使用多项式空间对趋势模型进行拟合
        trend_vals = trend_vals.transpose(1, 2)#转置趋势值，匹配输入的维度
        return trend_vals
    

class MovingBlock(nn.Module):#移动平均模型时间序列的趋势
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)#计算移动平均的窗口大小，确保再合理范围内
        self.decomp = series_decomp(size)#创造时间序列分解对象，用于对输入数据进行移动平均分解

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)#使用移动平均模型对输入数据进行趋势分解
        return x, trend_vals


class FourierLayer(nn.Module):#对时间序列进行傅里叶变换
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model#输入数据特征维度
        self.factor = factor#选择保留频率的因子
        self.low_freq = low_freq#傅里叶变化中保留的最低频率

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)#对输入数据进行实数域的快速傅里叶变换

        if t % 2 == 0:#长度是否偶数
            x_freq = x_freq[:, self.low_freq:-1]#将频率与数据切片
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]#根据序列长度生成对应的频率序列，并进行切片操作
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)#选择前k个频率，并返回选择后的频率表示和相应的索引元组
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)#将频率序列扩展成与数据匹配的形状，重复次数由数据的batchsize和特征维度决定，最终移动到相同设备上
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)#调整频率序列的形状以便进行后续运算，确保它与数据匹配
        return self.extrapolate(x_freq, f, t)#调用函数对频率进行外推，得到时间域表示的数据

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)#将频率和其共轭频率连接起来，形成一个包含正频率和负频率的频率表示
        f = torch.cat([f, -f], dim=1)#将频率序列和其负频率序列连接起来，确保包含正频率和负频率
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)#生成时间序列，保持与频率匹配的形状

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')#计算频率幅度，调整为与时间序列匹配的形状
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')#计算相位
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)#使用频率/幅度和相位计算时间序列表示
        return reduce(x_time, 'b f t d -> b t d', 'sum')#对所有频率的时间序列进行求和，得到最终时间序列表示

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]#快速傅里叶变换形状，频率序列长度等于时间序列长度的一半
        top_k = int(self.factor * math.log(length))#采用log函数的变换，选择保留前K个频率数目
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)#对每个样本的频率进行排序，选择出每个样本中绝对值最大的前k个频率
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')#生成用于选择前k个频率的索引元组。mesha，b是两个网格矩阵，用于与indices相结合形成索引元组
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))#将两个网格矩阵与频率索引结合，生成用于选择前K个频率的索引元组
        x_freq = x_freq[index_tuple]#选择前K个频率
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0#确保嵌入维度可以被注意力头数量整除
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)#创建三个线性变换层，分别用于对输入进行键/查询和值的投影

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)#创建两个dropout层，用于注意力权重和残差的正则化
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head#创建线性变换层用于输出投影，并保存注意力头的数量

    def forward(self, x, mask=None):
        B, T, C = x.size()#获取输入张量的形状信息，批次大小，序列长度，特征维度
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)，分别通过键。查询和值的线性变换层，对结果进行形状调整，以适应多头注意力机制的计算
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)，计算注意力分数，使用点积注意力机制，除以注意力头的维度进行缩放

        att = F.softmax(att, dim=-1) # (B, nh, T, T)，对注意力分数进行softmax操作，并应用注意力权重的dropout
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)，计算加权后的值
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)，将多头注意力的输出进行形状调整，以便后续进行输出投影
        att = att.mean(dim=1, keepdim=False) # (B, T, T)，计算注意力分数的均值

        # output projection
        y = self.resid_drop(self.proj(y))#对输出进行残差连接，并应用残差权重的dropout
        return y, att


class CrossAttention(nn.Module):#输入不同
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)，计算注意力分数，除以注意力头的维度进行缩放

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)#创建第一个LayerNorm模块，用于对输入序列进行层归一化。
        self.ln2 = nn.LayerNorm(n_embd)#创建第二个LayerNorm模块，用于对注意力层的输出进行层归一化。
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )#创建注意力层，用于对输入序列进行自注意力操作。
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )#创建一个多层感知机模块，用于对注意力层的输出进行全连接操作
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)#将输入序列经过第一个LayerNorm后传入注意力层，得到注意力输出和注意力权重
        x = x + a#将残差俩姐和注意力层的输出相加，得到编码器块的输出
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output，将编码器块的输出经过第二个LayerNorm和MLP模块后再次与原始输入相加，得到最终的编码器块输入
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,#transformer块的数量
        n_embd=1024,#嵌入维度
        n_head=16,#注意力头的数量
        attn_pdrop=0.,#注意力权重的dropout概率
        resid_pdrop=0.,
        mlp_hidden_times=4,#MLP隐藏层的倍数
        block_activate='GELU',#激活函数类型
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])#创建了一个由多个Transformer块组成的序列。每个Transformer块由参数指定的配置创建，并通过EncoderBlock类进行实例化。

    def forward(self, input, t, padding_masks=None, label_emb=None):#前向传播函数，定义了数据在编码器中的流动。input表示输入数据，t表示时间步，padding_masks表示填充的掩码，label_emb表示标签的嵌入
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x#对当前的输入数据x应用第block_idx个Transformer块，并将处理后的数据赋值给x。由于不需要返回注意力权重等信息，使用下划线_来接收这部分信息。


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )#自注意力机制模块，处理数据中的相关性
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )#交叉注意力机制模块，将编码器的输出与解码器的输入进行交互
        
        self.ln1_1 = AdaLayerNorm(n_embd)#自适应层归一化，对交叉注意力机制输出的数据进行归一化

        assert activate in ['GELU', 'GELU2']#确保激活函数类型合法性
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)#创建一个趋势模块，用于对输入数据中的趋势进行建模
        # self.decomp = MovingBlock(n_channel)
        #self.seasonal = FourierLayer(d_model=n_embd)#创建傅里叶季节模块，用于对输入数据中的季节性进行建模
        self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )#创建多层感知机模块，用于对输入数据进行非线性变换

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)#创建一个一维卷积层，用于将输入数据进行通道分离
        self.linear = nn.Linear(n_embd, n_feat)#创建线性层，将嵌入维度映射到特征数

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)#将处理后的序列进行投影并分成两部分，得到趋势和季节的处理
        trend, season = self.trend(x1), self.seasonal(x2)#将投影后的两部分分别经过趋势块和季节块的处理，得到趋势和季节部分
        x = x + self.mlp(self.ln2(x))#将处理后的序列经过多层感知机处理后与原始序列相加，得到最终序列
        m = torch.mean(x, dim=1, keepdim=True)#计算序列的均值
        return x - m, self.linear(m), trend, season#返回去均值后的序列，均值的线性变换结果
    

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512    
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
        ) for _ in range(n_layer)])#创建一个由n_layer个transformer块组成的序列，每个块由参数指定配置创建，并通过decoderblock进行实例化
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape#b批次大小，c通道数
        # att_weights = []
        mean = []#空列表储存transformer块输出的均值
        season = torch.zeros((b, c, self.d_model), device=x.device)#创建季节性特征
        trend = torch.zeros((b, c, self.n_feat), device=x.device)#创建趋势特征
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)#对当前的输入数据x应用第block_idx个transformer块，并将处理后的数据/均值/趋势和季节性特征赋值给对应的变量
            season += residual_season#将当前的transformer块输出的季节性特征累加到整体
            trend += residual_trend#累加
            mean.append(residual_mean)#均值添加到列表中

        mean = torch.cat(mean, dim=1)#列表所有均值张量拼接成一个张量，沿维度1及逆行拼接
        return x, mean, trend, season


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,#特征数
        n_channel,#通道数
        n_layer_enc=5,#编码器层数
        n_layer_dec=14,#解码器层数
        n_embd=1024,#嵌入维度
        n_heads=16,#注意力头的数量
        attn_pdrop=0.1,#注意力权重的dropout概率
        resid_pdrop=0.1,#残差注意力权重的dropout概率
        mlp_hidden_times=4,#MLP隐藏层的倍数
        block_activate='GELU',#激活函数类型
        max_len=2048,#序列最大长度
        conv_params=None,#卷积参数
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)#创建了一个编码器
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)#创建位置编码器，编码器的输入进行位置编码

        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)#创建解码器
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)#位置解码器

    def forward(self, input, t, padding_masks=None):#前向传播
        emb = self.emb(input)#对输入序列进行嵌入操作，将原始特征映射到嵌入空间
        inp_enc = self.pos_enc(emb)#嵌入序列进行位置编码
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)#对位置编码好的序列进行编码

        inp_dec = self.pos_dec(emb)#位置解码
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)#解码，得到解码后的输出，均值，趋势和季节特征

        res = self.inverse(output)#对解码器的输出进行逆变换，将嵌入空间的表示映射回原始特征空间
        res_m = torch.mean(res, dim=1, keepdim=True)#计算解码器输出的均值
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m#计算季节性误差
        trend = self.combine_m(mean) + res_m + trend#计算趋势特征

        return trend, season_error


if __name__ == '__main__':
    pass