
import torch
from einops import rearrange, reduce, repeat
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import math

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1)) #matmul,矩阵乘法   permute用于维度转换，原本为（0，1，2）进行换位即可
        if scale:
            attention = attention * scale
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
    
class crossAttention(nn.Module):
    def __init__(self, dim_model, num_head=4, dropout=0.0):
        super(crossAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head) #这相当于一个方阵
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)


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

        return out
    
class crossFrequencyAttention(nn.Module):
    def __init__(self, dim_model, num_head=1, dropout=0.0):
        super(crossFrequencyAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head) #这相当于一个方阵
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)


    def forward(self, x):
        batch_size = x.size(0)
        Q = torch.fft.rfft(self.fc_Q(x), dim=1)
        K = torch.fft.rfft(self.fc_K(x), dim=1)
        V = torch.fft.rfft(self.fc_V(x), dim=1)

        scale = K.size(-1) ** -0.5  # 缩放因子
        
        attention = torch.matmul(Q, K.permute(0, 2, 1)) #matmul,矩阵乘法   permute用于维度转换，原本为（0，1，2）进行换位即可
        if scale:
            attention = attention * scale
        attention = F.softmax(abs(attention), dim=-1)
        attention = torch.complex(attention, torch.zeros_like(attention ))
        context = torch.matmul(attention, V)
        
        context = torch.fft.irfft(context, dim=1, n=x.shape[-2])
        
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)

        return out
    
class temporalFrequencyAttention(nn.Module):
    def __init__(self, dim_model, num_head=1, dropout=0.0):
        super(temporalFrequencyAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head) #这相当于一个方阵
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1)
        Q = torch.fft.rfft(self.fc_Q(x), dim=1)
        K = torch.fft.rfft(self.fc_K(x), dim=1)
        V = torch.fft.rfft(self.fc_V(x), dim=1)

        scale = K.size(-1) ** -0.5  # 缩放因子
        
        attention = torch.matmul(Q, K.permute(0, 2, 1)) #matmul,矩阵乘法   permute用于维度转换，原本为（0，1，2）进行换位即可
        if scale:
            attention = attention * scale
        attention = F.softmax(abs(attention), dim=-1)
        attention = torch.complex(attention, torch.zeros_like(attention ))
        context = torch.matmul(attention, V)
        
        context = torch.fft.irfft(context, dim=1, n=x.shape[-2])
        
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)

        return out.permute(0,2,1)    
    
    
class crossFrequencyAttention1(nn.Module):
    def __init__(self, dim_model, num_head=1, dropout=0.0):
        super(crossFrequencyAttention1, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head) #这相当于一个方阵
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        
        index = list(range(0, 48 // 2))
        np.random.shuffle(index)
        self.index_q = index[:24]

    def forward(self, x):
        batch_size = x.size(0)
        B, H, E = x.shape[0], x.shape[1], x.shape[2]
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)

        xq_ft_ = torch.zeros(B, H, len(self.index_q), device=x.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(Q, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, i] = xq_ft[:, :, j]
        xk_ft_ = torch.zeros(B, H, len(self.index_q), device=x.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(K, dim=-1)
        for i, j in enumerate(self.index_q):
            xk_ft_[:, :, i] = xk_ft[:, :, j]
        xv_ft_ = torch.zeros(B, H, len(self.index_q), device=x.device, dtype=torch.cfloat)
        xv_ft = torch.fft.rfft(K, dim=-1)
        for i, j in enumerate(self.index_q):
            xv_ft_[:, :, i] = xv_ft[:, :, j]
            
        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhx,bhy->bxy", xq_ft_, xk_ft_))
        xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
        xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))

        xqkv_ft = torch.einsum("bxy,bey->bex", xqk_ft, xv_ft_)
        out_ft = torch.zeros(B, H, E// 2 + 1, device=x.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, j] = xqkv_ft[:, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft, n=x.size(-1))
        return out

class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')
    
    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    
class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            nn.GELU(),
            Transpose(shape=(1, 2)),
            # nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals

class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return trend_vals

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


class synTemporalBlock(nn.Module):
    def __init__(self, n_channel, n_embd):
        super(synTemporalBlock, self).__init__()
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_embd)
        # self.trend = MovingBlock(n_channel)
        # self.seasonal = FourierLayer(d_model=n_embd)
        self.seasonal = SeasonBlock(n_channel, n_channel)
    def forward(self, x):
        return self.trend(x) + self.seasonal(x)

class synTemporalBlock_ori(nn.Module):
    def __init__(self, n_channel, n_embd):
        super(synTemporalBlock_ori, self).__init__()
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_embd)
        # self.trend = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)
    def forward(self, x):
        return self.trend(x) + self.seasonal(x)
    
class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter = True
        nn.init.trunc_normal_(self.complex_weight_high, std=.02)
        nn.init.trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        
        
    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x_in):
        x_in = x_in.transpose(1,2)
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x.transpose(1,2)


if __name__ == '__main__':
    batch_size = 8
    input_size = 18
    window_size = 48
    # model = synTemporalBlock(18,48)
    model = Adaptive_Spectral_Block(18)
    x = torch.randn(batch_size, input_size, window_size)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size,1])
    y = model(x)
    print(y.shape)