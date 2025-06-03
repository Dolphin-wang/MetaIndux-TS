import torch
import data.CMAPSSDataset as CMAPSSDataset
from torch.utils.data import DataLoader,TensorDataset,random_split
import numpy as np
from utils import get_time_dif,score_calculate,adjust_learning_rate,adjust_learning_rate_decay,rmse,mae
import time
import torch.nn.functional as F
from models.LSTM  import Config,LSTM
from args import args
import wandb

def train(config, model, train_iter, dev_iter, test_iter=None):
    print('model_name:{}  learning_rate:{}  window_size:{}  embedding:{}  '.format(
        config.model_name, config.lr, config.window_size, config.embedding)) 
    start_time = time.time()
    best_epoch = 15
    best_loss = float('inf') # 正无穷
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    
    for epoch in range(config.eva_epoch):
        model.train()
        count,train_loss,dev_loss = 0,0,0
        lr = adjust_learning_rate_decay(optimizer,optimizer, epoch, config.lr)
        print('\nEpoch [{}/{}]'.format(epoch + 1, config.eva_epoch))
        for i, (trains, labels) in enumerate(train_iter):
            model.train()
            outputs = model(trains)
            model.zero_grad()
            loss = F.mse_loss(outputs, labels) 
            train_loss += loss
            loss.backward()
            optimizer.step()
            count += 1
            if count % (len(train_iter) // 5) == 0:                
                train_loss = train_loss.detach().cpu()/(len(train_iter) // 5)
                dev_loss, _, _ =  evaluate( model, dev_iter)
                test_loss, mae, score_total = evaluate( model, test_iter)
                time_dif = get_time_dif(start_time)
                print('Learning Rate:{:.3e}   Train_Loss:{:.3f}   Dev_loss:{:.3f}   Test_Loss:{:.3f}   Score:{:.3f}   Time:{}'.format(lr, np.sqrt(train_loss),\
                                                                                                            dev_loss, test_loss, score_total, time_dif) )
                wandb.log({"Reg_train_Loss":train_loss, "Reg_test_Loss":test_loss})
                train_loss = 0  
                              
                if dev_loss < best_loss and epoch > 5:
                    best_loss = dev_loss
                    best_epoch = epoch
                    eva_test_loss,eva_score,eva_mae = test_loss, score_total, mae 
                    torch.save(model, config.save_path)
                    print('*******imporove!!!********')
        if epoch - best_epoch >= 10:
            print('*******STOP!!!********')
            break      
    print('best_test_loss:{}  best_score:{}  '.format(eva_test_loss,eva_score)) 
    return eva_test_loss, eva_mae, eva_score
    
def evaluate(model, data_iter):
    model.eval()
    loss_total,score = 0, 0 
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.mse_loss(outputs, labels)
            loss_total += loss
            labels = labels.detach().cpu()
            predic = outputs.cpu()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    score = score_calculate(predict_all, labels_all)
    rmse1 = rmse(predict_all,labels_all)
    mae1 = mae(predict_all,labels_all)
    return rmse1, mae1, score / len(predict_all)   
    
def predictive_score_metrics(config, ori_data, generated_data):
    # shape = ori_data.shape
    # ori_data,generated_data = torch.tensor(ori_data), torch.tensor(generated_data)
    train_data,train_label =  torch.tensor(generated_data['data']),  torch.tensor(generated_data['label'])
    test_data,test_label = ori_data['data'].clone().detach(),  ori_data['label'].clone().detach()

    print("train_data.shape",train_data.shape)
    print("test_data.shape",test_data.shape)  

    if train_data.shape[-1] == 21:
        train_data = train_data[:,:,[1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]]
    if test_data.shape[-1] == 21:
        test_data = test_data[:,:,[1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]]
    
     
    train_dataset = TensorDataset(train_data.to(config.device), train_label.to(config.device))
    data_len = len(train_dataset)
    train_dataset, dev_dataset = random_split(train_dataset, [int(data_len*0.8), data_len - int(data_len*0.8)])

    train_iter = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    dev_iter = DataLoader(dataset=dev_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(test_data.to(config.device), test_label.to(config.device))
    test_iter = DataLoader(dataset=test_dataset, batch_size=2048, shuffle=False)    
    
    config.output_size = 1
    model = LSTM(config).to(config.device)
    best_test_loss, mae, best_score = train(config, model, train_iter, dev_iter, test_iter)
    return best_test_loss,mae, best_score

if __name__ == '__main__':
    # wandb.init()
    args.dataset = 'FD001'
    loaded_data = np.load('./weights/syn_data/syn_'+ args.dataset+'_'+ args.model_name+'_'+str(48) +'.npz')
    datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=48,deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_feature_slice(train_data), datasets.get_label_slice(train_data)
    predictive_score_metrics(args, {'data':train_data,'label':train_label}, loaded_data)
