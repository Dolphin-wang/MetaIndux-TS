
import torch
import time
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset,random_split
import numpy as np
from DiffusionFreeGuidence.Unet1D import UNet1D
from DiffusionFreeGuidence.Unet1D_fre import UNet1D_fre
from DiffusionFreeGuidence.Diffwave import DiffWave
from DiffusionFreeGuidence.tabddpm import MLPDiffusion
from DiffusionFreeGuidence.Dit import DiT
from DiffusionFreeGuidence.SSSD import SSSDS4Imputer

from DiffusionFreeGuidence.diffwaveimputer import DiffWaveImputer
from DiffusionFreeGuidence.csdi import diff_CSDI

import data.CMAPSSDataset as CMAPSSDataset
import wandb
from GaussianDiffusion import GaussianDiffusion1D_cls_free,GradualWarmupScheduler

def train(args, train_data, train_label):
    device = args.device
    best_loss = 9999

    train_dataset = TensorDataset(train_data.permute(0,2,1).to(device), train_label.to(device))
    dataloader= DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    
    # model setup
    if args.model_name == 'DiffUnet': 
        net_model = UNet1D(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size).to(device) #cmapss  dim = 32 dim_mults = (1, 2, 2) cond_drop_prob = 0.5
    if args.model_name == 'DiffUnet_fre': 
        net_model = UNet1D_fre(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device)  
    if args.model_name == 'dit': 
        net_model = DiT(input_size=args.input_size,hidden_size=128, num_heads=1).to(device) 
    if args.model_name == 'DiffWave': 
        net_model = DiffWaveImputer(seq_length=args.window_size)
    if args.model_name == 'SSSD': 
        net_model = SSSDS4Imputer(seq_length=args.window_size, s4_lmax=args.window_size )
    if args.model_name == 'tabddpm':     
        net_model = MLPDiffusion( d_in=14, num_classes=0, is_y_cond=True, window_size=args.window_size, rtdl_params={'d_layers': [128, 256, 256, 128], 'dropout': 0.3}, dim_t = 512)
    if args.model_name == 'csdi': 
        net_model = diff_CSDI(inputdim=14)   
                 
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=args.multiplier,
                                             warm_epoch=args.epoch // 10 + 1, after_scheduler=cosineScheduler)
    print(args.T)
    trainer = GaussianDiffusion1D_cls_free(net_model, seq_length=args.window_size, channels = args.input_size, timesteps =  args.T, objective='pred_noise').to(device)

    # start training
    for e in range(args.epoch):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            loss_list=[]
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, classes = labels).sum()
                loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), args.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": sum(loss_list)/len(loss_list),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        current_loss = sum(loss_list)/len(loss_list)
        wandb.log({"Diffusion_Loss":current_loss})
        if e > 5 and current_loss < best_loss:
            torch.save(net_model.state_dict(), args.model_path)
            print('*******imporove!!!********')

def sample(args, train_label = None):
    if train_label == None:
        datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size ,deleted_engine=[1000])
        train_data = datasets.get_train_data()
        train_label = datasets.get_label_slice(train_data)        
    device = args.device
    # load model and evaluate
    with torch.no_grad():
        if args.model_name == 'DiffUnet': 
            net_model = UNet1D(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size).to(device) #cmapss  dim = 32 dim_mults = (1, 2, 2) cond_drop_prob = 0.5
        if args.model_name == 'DiffUnet_fre': 
            net_model = UNet1D_fre(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
        if args.model_name == 'DiffWave': 
            net_model = DiffWaveImputer(seq_length=args.window_size)
        if args.model_name == 'SSSD': 
            net_model = SSSDS4Imputer(seq_length=args.window_size, s4_lmax=args.window_size )
        if args.model_name == 'tabddpm':     
            net_model = MLPDiffusion( d_in=14, num_classes=0, is_y_cond=True, window_size=args.window_size, rtdl_params={'d_layers': [128, 256, 256, 128], 'dropout': 0.0}, dim_t = 512)
        if args.model_name == 'csdi':     
            net_model = diff_CSDI(inputdim=14)
        if args.model_name == 'dit': 
            net_model = DiT(input_size=args.input_size,hidden_size=128, num_heads=1).to(device)                               
        ckpt = torch.load(args.model_path)
        net_model.load_state_dict(ckpt)
        print("model load weight done.")
        net_model.eval()
        diffusion = GaussianDiffusion1D_cls_free(net_model, seq_length=args.window_size, channels = args.input_size, timesteps =  args.T, objective='pred_noise').to(device)
        # Sampled from standard normal distribution
        start_time = time.time()
        sampledata = diffusion.sample(classes = train_label.to(device)).permute(0,2,1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"经过的时间: {elapsed_time} 秒")
        
        sampledata = sampledata.cpu().numpy()
        np.savez(args.syndata_path,data=sampledata, label =  train_label.cpu().numpy())      

        #np.savez(args.syndata_path,data=sampledata.cpu().numpy(), label =  train_label.cpu().numpy())
    return sampledata