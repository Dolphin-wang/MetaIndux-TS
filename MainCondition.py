import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from TrainCondition import train, sample
import sys
from data.CMAPSSDataset import CMAPSSDataset
import wandb
from utils import wandb_record,torch_seed
from eva_regressor import predictive_score_metrics
from data_process import load_train_data,load_test_data,load_test_data_rul,load_train_data_rul
from measure_score.Utils.discriminative_metric import discriminative_score_metrics
from measure_score.Utils.context_fid import Context_FID
from measure_score.Utils.cross_correlation import CrossCorrelLoss

from args import args
import torch

os.environ["WANDB_MODE"] = "offline"
rmse_list,score_list,acc_list, FID_list, CorrelLoss_list, mae_list = [],[],[],[],[],[]


if __name__ == '__main__':
    if len(sys.argv)==1:
        print('-------no prompt--------')
        args.epoch = 50
        args.dataset = 'FD001'
        args.lr = 2e-3
        args.state = 'sample' # all,train,sample,eval
        args.model_name = 'DiffUnet_fre' 
        args.T = 1000
        args.window_size = 48
        args.w = 0
        args.input_size = 14
    wandb.init(project="MetaIndux-TS", tags=['all'], config=args )
    train_loop = 1
    torch_seed(3)
    args.model_path =  'weights/' + args.model_name + '_' + args.dataset + '_' + str(args.window_size) + '.pth'
    args.syndata_path =  './weights/syn_data/syn_'+ args.dataset+'_'+args.model_name + '_' + str(args.window_size) + args.sample_type +'.npz'

    datasets = CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size, deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_feature_slice(train_data), datasets.get_label_slice(train_data)
    
    test_data = datasets.get_test_data()
    test_data,test_label = datasets.get_last_data_slice(test_data)
    
    train_data,train_label = train_data[0:len(train_data)], train_label[0:len(train_label)]
    print("train_data.shape:",train_data.shape,"      test_data.shape:",test_data.shape)
    
    
    if args.state == "train" or args.state == "all":
        train(args,train_data,train_label)
        sample(args,train_label)
    elif args.state == "sample":
        sample(args,train_label)
    if args.state == "eval" or args.state == "all" or args.state == "sample":
        syn_dataset = np.load(args.syndata_path)
        syn_data = syn_dataset['data']
        original_data_test = {'data':test_data,'label':test_label}
        original_data_train = {'data':train_data,'label':train_label}
        concat_data = {}
        random_indices = np.random.choice(train_data.shape[0], size=len(train_data) // 10 , replace=False)

        for i in range(train_loop):
            rmse,mae, score= predictive_score_metrics(args, original_data_test, syn_dataset)
            discriminative_score, fake_accuracy, real_accuracy = discriminative_score_metrics(train_data.cpu().numpy() , syn_data)
            Context_FID_score = Context_FID(train_data.cpu().numpy(), syn_data) # Context_FID分数计算
            print("Context_FID_score:", Context_FID_score)

            loss_function = CrossCorrelLoss(train_data.cpu().numpy(), name=args.dataset) # Correlation分数计算
            CrossCorrel_Loss = loss_function(torch.tensor(syn_data))
            rmse_list.append(rmse); score_list.append(score); acc_list.append(discriminative_score)
            mae_list.append(mae); FID_list.append(Context_FID_score); CorrelLoss_list.append(CrossCorrel_Loss)

        wandb_record(rmse_list,mae_list,score_list, acc_list,FID_list,CorrelLoss_list)
        wandb.finish()

