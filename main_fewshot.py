import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from TrainCondition import train, sample
import sys
from data.CMAPSSDataset import CMAPSSDataset
import wandb
from utils import wandb_record,torch_seed
from eva_regressor import predictive_score_metrics
from eva_classifier import discrimative_score_metrics
from data_process import load_train_data,load_test_data,load_test_data_rul,load_train_data_rul
from measure_score.Utils.discriminative_metric import discriminative_score_metrics
from measure_score.Utils.context_fid import Context_FID
from measure_score.Utils.cross_correlation import CrossCorrelLoss
#from data.data_process import load_RUL2012

from args import args
import torch

os.environ["WANDB_MODE"] = "offline"
rmse_list,score_list,acc_list, FID_list, CorrelLoss_list, mae_list = [],[],[],[],[],[]


if __name__ == '__main__':
    if len(sys.argv)==1:
        print('-------no prompt--------')
        args.epoch = 70
        args.dataset = 'FD002'
        args.lr = 2e-3
        args.state = 'sample' # all,train,sample,eval
        args.model_name = 'DiffUnet' 
        args.T = 180
        args.window_size = 48
        args.w = 0
        args.input_size = 14
        args.task = 'fewshot'
    if args.model_name == 'dit':
        wandb.init(project="DiffFre", tags=['EXP-compare'], config=args )
    else:
        wandb.init(project="DiffFre", tags=['Fre-w/o_syn'], config=args )
    train_loop = 5
    torch_seed(5)
    args.model_path =  'weights/' + args.model_name + '_' + args.dataset + '_' + str(args.window_size) + args.task + '.pth'
    args.syndata_path =  './weights/syn_data/syn_'+ args.dataset+'_'+args.model_name + '_' + str(args.window_size) + args.sample_type + args.task +'.npz'

    datasets = CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size, deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_feature_slice(train_data), datasets.get_label_slice(train_data)
    
    test_data = datasets.get_test_data()
    test_data,test_label = datasets.get_last_data_slice(test_data)

    train_data,train_label = train_data[0:len(train_data)], train_label[0:len(train_label)]
    if args.task == 'zeroshot':
        mask1 = train_label.squeeze(-1) > 30  
        filtered_train_data = train_data[mask1]
        filtered_train_label = train_label[mask1]
    elif args.task == 'fewshot':
        # 找到大于30的索引
        mask_greater = train_label.squeeze(-1) > 30
        filtered_train_data = train_data[mask_greater]
        filtered_train_label = train_label[mask_greater]
        # 找到小于等于30的索引
        mask_smaller = train_label.squeeze(-1) <= 30
        small_data = train_data[mask_smaller]
        small_label = train_label[mask_smaller]
        # 计算 5% 的采样数量
        sample_size = int(0.1 * small_data.shape[0])
        # 随机采样 5% 的小于等于30的数据
        if sample_size > 0:
            sampled_indices = torch.randperm(small_data.shape[0])[:sample_size]
            sampled_train_data = small_data[sampled_indices]
            sampled_train_label = small_label[sampled_indices]
            # 合并数据
            filtered_train_data = torch.cat([filtered_train_data, sampled_train_data], dim=0)
            filtered_train_label = torch.cat([filtered_train_label, sampled_train_label], dim=0)

    mask2 = test_label.squeeze(-1) <= 30  
    filtered_test_data = test_data[mask2]
    filtered_test_label = test_label[mask2]
    if args.task != 'all':
        print("train_data.shape:",filtered_train_data.shape,"      test_data.shape:", filtered_test_data.shape)
    print("train_data.shape:",train_data.shape,"      test_data.shape:",test_data.shape)
    
    
    if args.state == "train" or args.state == "all":
        if args.task == 'all':
            train(args,train_data,train_label)
        else:
            train(args,filtered_train_data,filtered_train_label)
        sample(args,train_label)
    elif args.state == "sample":
        sample(args,train_label)
    if args.state == "eval" or args.state == "all" or args.state == "sample":
        syn_dataset = np.load(args.syndata_path)
        syn_data = syn_dataset['data']
        #original_data_test = {'data':test_data,'label':test_label}
        filtered_data_test = {'data':filtered_test_data,'label':filtered_test_label}
        original_data_train = {'data':train_data,'label':train_label}
        concat_data = {}
        random_indices = np.random.choice(train_data.shape[0], size=len(train_data) // 10 , replace=False)

        for i in range(train_loop):
            #rmse,mae, score= predictive_score_metrics(args, original_data_test, syn_dataset) #syn_dataset)
            rmse,mae, score= predictive_score_metrics(args, filtered_data_test, syn_dataset) #syn_dataset)
            #discriminative_score, fake_accuracy, real_accuracy = discriminative_score_metrics(train_data.cpu().numpy() , syn_data)
            # discriminative_score,Context_FID_score = 0,0
            '''Context_FID_score = Context_FID(train_data.cpu().numpy(), syn_data) # Context_FID分数计算
            print("Context_FID_score:", Context_FID_score)

            loss_function = CrossCorrelLoss(train_data.cpu().numpy(), name=args.dataset) # Correlation分数计算
            CrossCorrel_Loss = loss_function(torch.tensor(syn_data))
            print("损失值:", CrossCorrel_Loss.item())
            # rmse,score,discriminative_score, mae,  Context_FID_score = 0,0,0,0,0'''
            rmse_list.append(rmse)
            #rmse_list.append(rmse); score_list.append(score); acc_list.append(discriminative_score)
            #mae_list.append(mae); FID_list.append(Context_FID_score); CorrelLoss_list.append(CrossCorrel_Loss)
        with open('output_task.txt', 'a') as f:
            f.write("\n\n--- New Data Entry ---\n")
            f.write(f"Model_name: {args.model_name}\n")
            f.write(f"Dataset: {args.dataset}\n")
            #f.write(f"ArgK: {args.argk}\n")
            f.write(f"task: {args.task}\n")
            f.write("RMSE Values:\n")
            for value in rmse_list:
                f.write(f"{value}\n")

        #print("loss_list",rmse_list,"acc list",acc_list)
        #wandb_record(rmse_list,mae_list,score_list, acc_list,FID_list,CorrelLoss_list)
        #wandb.finish()

