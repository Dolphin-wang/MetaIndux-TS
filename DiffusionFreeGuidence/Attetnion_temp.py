
import torch
from einops import rearrange, reduce, repeat
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import math


    
class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
    
        
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

        self.seasonal = SeasonBlock(n_channel, n_channel)
    def forward(self, x):
        return self.trend(x) + self.seasonal(x)

class Adaptive_Spectral_Block_t(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter = True
        nn.init.trunc_normal_(self.complex_weight_high, std=.02)
        nn.init.trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        
        dim_model = dim
        num_head = 1
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head) #这相当于一个方阵
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)

        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        
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
        
        Q = x_fft
        K = torch.fft.rfft(self.fc_K(x), dim=1, norm='ortho')
        V = torch.fft.rfft(self.fc_V(x), dim=1, norm='ortho')

        scale = K.size(-1) ** -0.5  # 缩放因子
        
        attention = torch.matmul(Q, K.permute(0, 2, 1)) #matmul,矩阵乘法   permute用于维度转换，原本为（0，1，2）进行换位即可
        if scale:
            attention = attention * scale
        attention = F.softmax(abs(attention), dim=-1)
        attention = torch.complex(attention, torch.zeros_like(attention ))
        context = torch.matmul(attention, V)

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            context += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(context, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x.transpose(1,2)

class Adaptive_Spectral_Block_c(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter = True
        nn.init.trunc_normal_(self.complex_weight_high, std=.02)
        nn.init.trunc_normal_(self.complex_weight, std=.02)
        
        dim_model = dim
        num_head = 1
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head) #这相当于一个方阵
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        

    def forward(self, x_in):
        x_in = x_in.transpose(1,2)
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        Q = x_fft.permute(0, 2, 1)
        K = torch.fft.rfft(self.fc_K(x), dim=1).permute(0, 2, 1)
        V = torch.fft.rfft(self.fc_V(x), dim=1).permute(0, 2, 1)

        scale = K.size(-1) ** -0.5  # 缩放因子
        
        attention = torch.matmul(Q, K.permute(0, 2, 1)) #matmul,矩阵乘法   permute用于维度转换，原本为（0，1，2）进行换位即可
        if scale:
            attention = attention * scale
        attention = F.softmax(abs(attention), dim=-1)
        attention = torch.complex(attention, torch.zeros_like(attention ))
        context = torch.matmul(attention, V).permute(0, 2, 1)

        # Apply Inverse FFT
        x = torch.fft.irfft(context, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x.transpose(1,2)

class Adaptive_Spectral_Block_c_old(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter = True
        nn.init.trunc_normal_(self.complex_weight_high, std=.02)
        nn.init.trunc_normal_(self.complex_weight, std=.02)
        
        dim_model = dim
        num_head = 1
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head) #这相当于一个方阵
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        

    def forward(self, x_in):
        x_in = x_in.transpose(1,2)
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        Q = x_fft
        K = torch.fft.rfft(self.fc_K(x), dim=1)
        V = torch.fft.rfft(self.fc_V(x), dim=1)

        scale = K.size(-1) ** -0.5  # 缩放因子
        
        attention = torch.matmul(Q, K.permute(0, 2, 1)) #matmul,矩阵乘法   permute用于维度转换，原本为（0，1，2）进行换位即可
        if scale:
            attention = attention * scale
        attention = F.softmax(abs(attention), dim=-2)
        attention = torch.complex(attention, torch.zeros_like(attention ))
        context = torch.matmul(attention, V)

        # Apply Inverse FFT
        x = torch.fft.irfft(context, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x.transpose(1,2)
    
        
if __name__ == '__main__':
    batch_size = 8
    input_size = 18
    window_size = 48
    # model = synTemporalBlock(18,48)
    model = Adaptive_Spectral_Block_c1(18)
    x = torch.randn(batch_size, input_size, window_size)
    t = torch.randint(1000, size=[batch_size])
    labels = torch.randint(10, size=[batch_size,1])
    y = model(x)
    print(y.shape)