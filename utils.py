import numpy as np
import os 
import pickle as pkl
from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pandas as pd
import math
import wandb

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device) -> None:
        self.batch_size = batch_size
        self.batches = batches
        self.batch_num = len(batches[0]) // batch_size
        self.residue = False
        if len(batches) % batch_size != 0: #记录是否为小数，true为小数
            self.residue = True
        self.index = 0
        self.device = device
    
    def _to_tensor(self, datas):
        x = datas[0].to(self.device)
        y = datas[1].to(self.device) #这里要注意，转化为tensor变量

        return (x, y)

    def __next__(self):
        if  self.residue and self.index == self.batch_num:
            data_batches = self.batches[0][self.index * self.batch_size : ]
            label_batches = self.batches[1][self.index * self.batch_size : ]
            batches = (data_batches, label_batches)
            self.index += 1
            return self._to_tensor(batches)
        elif self.index >= self.batch_num:
            self.index = 0
            raise StopIteration
        else :
            data_batches = self.batches[0][self.index * self.batch_size : (self.index + 1) * self.batch_size]
            label_batches = self.batches[1][self.index * self.batch_size : (self.index + 1) * self.batch_size]
            batches = (data_batches, label_batches)
            self.index += 1
            batches = self._to_tensor(batches)

            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.batch_num + 1
        else:
            return self.batch_num    

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter
    
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def score_calculate(pred, true):
    """计算检测得分"""
    sum = 0
    for i in range(len(pred)):
        if pred[i] >= true[i]:
            sum += math.exp((pred[i]-true[i])/10) - 1
        else:
            sum += math.exp((true[i]-pred[i])/13) - 1
    return sum 


def adjust_learning_rate(optimizer1, optimizer2, epoch, lr):
    """调整学习率"""
    warmup_epoch = 30
    if epoch <= warmup_epoch:
        lr = lr * (epoch+5) / (warmup_epoch+5)
    if epoch >= warmup_epoch:
        lr = lr * ((warmup_epoch+5) / (epoch+5)) ** 3
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_decay(optimizer1, optimizer2, epoch, lr):
    """调整学习率"""
    lr, decay_rate, decay_step = lr, 0.1, [15,25,35]
    for i in decay_step:
        if epoch >= i:
            lr *= decay_rate
        else:
            break
        
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr 
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr 
    return lr

def torch_seed(seed_num = 2):
    np.random.seed(seed_num)
    torch.manual_seed(seed_num )
    torch.cuda.manual_seed(seed_num )#为cpu设置种子
     # 保证每次结果一样
    # torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mae(predictions, targets):
    return np.abs(predictions - targets).mean()

def wandb_record(rmse, mae, score, discri_score, Context_FID, CorrelLoss):
    metrics = {
        "eva_rmse": np.mean(np.array(rmse)),
        "eva_mae": np.mean(np.array(score)),
        "eva_score": np.mean(np.array(mae)),
        "eva_discri_score":np.mean(np.array(discri_score)),
        "eva_Context_FID":np.mean(np.array(Context_FID)),
        "eva_CorrelLoss":np.mean(np.array(CorrelLoss)),
        
        "eva_rmse_std": np.std(np.array(rmse)),
        "eva_mae_std": np.std(np.array(score)),
        "eva_score_std": np.std(np.array(mae)),
        "eva_discri_score_std":np.std(np.array(discri_score)),
        "eva_Context_FID_std":np.std(np.array(Context_FID)),
        "eva_CorrelLoss_std":np.std(np.array(CorrelLoss))
    }

    # 将指标更新到Weights & Biases的运行摘要中
    for key, value in metrics.items():
        wandb.run.summary[key] = value    
