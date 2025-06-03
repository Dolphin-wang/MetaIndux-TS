import numpy,torch
import torch.nn.functional as F
a = torch.tensor([[1,2,3],[1,2,3]]).to(torch.float32)
print(a.shape)
attention = F.softmax(abs(a), dim=-2)
print(attention)