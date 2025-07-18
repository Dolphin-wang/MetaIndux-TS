import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class SpectrogramUpsampler(nn.Module):
  def __init__(self, residual_channels, window_size):
    super().__init__()
    self.linear1 = Linear(1, window_size)
    self.linear2= Linear(1, residual_channels)

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.linear1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.linear2(x.permute(0,2,1)).permute(0,2,1)
    x = F.leaky_relu(x, 0.4)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, residual_channels, dilation, uncond=False):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    if not uncond: # conditional model
      self.conditioner_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
    else: # unconditional model
      self.conditioner_projection = None

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner=None):
    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    if self.conditioner_projection is None: # using a unconditional model
      y = self.dilated_conv(y)
    else:
      conditioner = self.conditioner_projection(conditioner)
      y = self.dilated_conv(y) + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
  def __init__(self, residual_channels=64,window_size=48, channels = 14 ,dilation_cycle_length=10,
               unconditional=True,residual_layers=5, time_step=50):
    super().__init__()
    self.unconditional = unconditional
    self.input_projection = Conv1d(channels, residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(time_step)
    if self.unconditional: # use unconditional model
      self.spectrogram_upsampler = None
    else:
      self.spectrogram_upsampler = SpectrogramUpsampler(residual_channels,window_size)

    self.residual_layers = nn.ModuleList([
        ResidualBlock(residual_channels, 2**(i % dilation_cycle_length), uncond=unconditional)
        for i in range(residual_layers)
    ])
    self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
    self.output_projection = Conv1d(residual_channels, channels, 1)
    nn.init.zeros_(self.output_projection.weight)
  def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
      
  def forward(self, audio, diffusion_step, spectrogram=None):
    assert (spectrogram is None and self.spectrogram_upsampler is None) or \
           (spectrogram is not None and self.spectrogram_upsampler is not None)

    x = audio
    x = self.input_projection(x)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    if self.spectrogram_upsampler: # use conditional model
      spectrogram = self.spectrogram_upsampler(spectrogram)

    skip = None
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, spectrogram)
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self

if __name__ == '__main__':
    batch_size = 8
    params = AttrDict(
        # Training 
        window_size = 48,
        # Data         n_mels=80,
        n_fft=1024,
        hop_samples=256,
        crop_mel_frames=62,  # Probably an error in paper.

        # Model         residual_layers=30,
        residual_channels=64,
        dilation_cycle_length=10,
        unconditional = False ,
        noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
        inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

        # unconditional sample len
        audio_len = 22050*5, # unconditional_synthesis_samples
    )
    
    model = DiffWave()
    x = torch.randn(batch_size,14, 48)
    t = torch.randint(50, size=[batch_size])
    labels = torch.randint(10, size=[batch_size,1]).float()
    # resB = ResBlock(128, 256, 64, 0.1)
    # x = torch.randn(batch_size, 128, 32, 32)
    # t = torch.randn(batch_size, 64)
    # labels = torch.randn(batch_size, 64)
    # y = resB(x, t, labels)
    y = model(x, t, labels)
    print(y.shape)
