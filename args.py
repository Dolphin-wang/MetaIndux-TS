import time
import argparse
import torch
arg_parser = argparse.ArgumentParser(description='RANet Image classification')



# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--model_name', type=str, default='DiffUnet')

# msdnet config
arch_group.add_argument('--embedding', type=int, default=48)
arch_group.add_argument('--hidden', type=int, default=64)

arch_group.add_argument('--num_head', type=int, default=1)
arch_group.add_argument('--num_encoder',type=int, default=1)

#TFS config
arch_group.add_argument('--lstm_hidden',type=int, default=64)
arch_group.add_argument('--num_layers',type=int, default=1)



# training related
optim_group = arg_parser.add_argument_group('optimization', 'optimization setting')
optim_group.add_argument('--epoch', default=50, type=int, metavar='N')
optim_group.add_argument('--eva_epoch', default=35, type=int, metavar='N')

optim_group.add_argument('-b', '--batch-size', default=64, type=int,metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='adam', choices=['sgd', 'rmsprop', 'adam'], metavar='N',)
optim_group.add_argument('--lr', '--learning_rate', default=2e-3, type=float, metavar='LR',)
optim_group.add_argument('--lr_type', default='multistep', choices=['multistep', 'cosine','warmup'], metavar='LR',)
optim_group.add_argument('--grad_clip',default=1.,type=float)
optim_group.add_argument('--multiplier',default=2.5,type=float)
optim_group.add_argument('--schedule_name',default='linear',type=str)
optim_group.add_argument('--task',default='all',type=str)
optim_group.add_argument('--loss_type',default='mse',choices=['mse', 'mse+mmd'],type=str)
optim_group.add_argument('--sample_type',default='ddim',choices=['ddim', 'ddpm'],type=str)

arg_parser.add_argument('--input_size',default=14,type=int)
arg_parser.add_argument('--output_size',default=1,type=int)
arg_parser.add_argument('--window_size',default=48,type=int)
arg_parser.add_argument('--dropout',default=0.05,type=float)
arg_parser.add_argument('--arch',default='att+td',choices=['att', 'original', 'att+td', 'td'],type=str)

arg_parser.add_argument('--dataset',default='FD001',type=str)
arg_parser.add_argument('--state',default='train',type=str)

arg_parser.add_argument('--T',default=500,type=int)
arg_parser.add_argument('--w',default=1.8,type=float)

arg_parser.add_argument('--device',default=torch.device('cuda' if torch.cuda.is_available() else 'cpu' ),type=str)
arg_parser.add_argument('--model_path',default='./weights/temp.pth',type=str)
arg_parser.add_argument('--save_path',default='./weights/lstm_temp.pth',type=str)
arg_parser.add_argument('--syndata_path',default='./weights/syn_data/temp.npy',type=str)

args = arg_parser.parse_args()
