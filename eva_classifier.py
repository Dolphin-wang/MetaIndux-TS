import torch
import data.CMAPSSDataset as CMAPSSDataset
from torch.utils.data import DataLoader,TensorDataset,random_split,ConcatDataset
import numpy as np
from utils import get_time_dif,score_calculate,adjust_learning_rate,adjust_learning_rate_decay,rmse
import time
import torch.nn.functional as F
from args import args
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.LSTM  import Config,LSTMClassifier,LSTM

def train(config, model, train_iter, dev_iter, test_iter=None):
    print('model_name:{}  learning_rate:{}  window_size:{}  embedding:{}  '.format(
        config.model_name, config.lr, config.window_size, config.embedding)) 
    start_time = time.time()
    best_epoch = 15
    best_acc = 0 # 正无穷
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    
    for epoch in range(config.eva_epoch):
        model.train()
        count,train_loss,running_correct = 0,0,0
        lr = adjust_learning_rate_decay(optimizer, optimizer, epoch, config.lr)
        print('\nEpoch [{}/{}]'.format(epoch + 1, config.eva_epoch))
        for i, (trains, labels) in enumerate(train_iter):
            model.train()
            outputs = model(trains)
            _, pred = torch.max(outputs.data, 1) # 第一个返回值是张量中最大值，第二个返回值是最大值索引
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels.squeeze(-1)) 
            train_loss += loss
            loss.backward()
            optimizer.step()
            count += 1
            if count % (len(train_iter) // 5) == 0:                
                train_loss = train_loss.detach().cpu()/(len(train_iter) // 5)
                dev_acc, test_acc =  evaluate(model, dev_iter),evaluate(model, test_iter)
                time_dif = get_time_dif(start_time)
                print('Learning Rate:{:.3e}   Train_Loss:{:.3f}   Dev_acc:{:.3f}   Test_acc:{:.3f}   Time:{}'.format(lr, np.sqrt(train_loss),\
                                                                                                             dev_acc, test_acc, time_dif) )
                wandb.log({"cla_train_Loss":dev_acc, "cla_test_acc":test_acc})
                train_loss = 0

    best_test_acc = evaluate(model, test_iter)
    print('best_test_acc:{} '.format(best_test_acc)) 
    return best_test_acc
    
def evaluate(model, data_iter):
    model.eval()
    testing_correct = 0
    labels_all,pred_all = np.array([], dtype=int),np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            _, pred = torch.max(outputs, 1)
            testing_correct += torch.sum(pred == labels.squeeze(-1))
            labels_all = np.append(labels_all, labels.squeeze(-1).cpu())
            pred_all = np.append(pred_all, pred.cpu())

    accuracy, precision, recall, f1 = accuracy_score(labels_all,pred_all), precision_score(labels_all,pred_all,zero_division=1.0), \
                                        recall_score(labels_all,pred_all), f1_score(labels_all,pred_all)
    return testing_correct.cpu().numpy() / len(labels_all)   
    
def discrimative_score_metrics(config, ori_data, generated_data):
    # shape = ori_data.shape
    
    train_data,train_label =  torch.from_numpy(generated_data['data']) , torch.ones_like(torch.from_numpy(generated_data['label']))
    test_data,test_label = ori_data['data'].clone().detach(),  torch.zeros_like(ori_data['label'].clone().detach())

    if train_data.shape[-1] == 21:
        train_data = train_data[:,:,[1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]]
    if test_data.shape[-1] == 21:
        test_data = test_data[:,:,[1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]]
        
    config.input_size= train_data.shape[2]

    train_dataset = TensorDataset(train_data.to(config.device), train_label.to(torch.long).to(config.device))
    test_dataset = TensorDataset(test_data.to(config.device), test_label.to(torch.long).to(config.device))
    concat_dataset = ConcatDataset([train_dataset, test_dataset])
    
    data_len = len(concat_dataset)
    split_sizes = [int(data_len * 0.7), int(data_len * 0.1), data_len - int(data_len * 0.7) - int(data_len * 0.1)]
    train_dataset, dev_dataset, test_dataset = random_split(concat_dataset, split_sizes)

    train_iter = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    dev_iter = DataLoader(dataset=dev_dataset, batch_size=64, shuffle=True)

    test_iter = DataLoader(dataset=test_dataset, batch_size=2048, shuffle=False)    
    
    config.output_size = 2
    # config.lstm_hidden,config.hidden = 2,2
    model = LSTMClassifier(config).to(config.device)
    test_acc = train(config, model, train_iter, dev_iter, test_iter)  
    return test_acc

if __name__ == '__main__':
    wandb.init()
    args.dataset = 'FD001'
    args.lr = 2e-4
    loaded_data = np.load('./weights/syn_data/syn_'+ args.dataset+'_'+ args.model_name +'.npz')
    datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=48,deleted_engine=[1000])
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_feature_slice(train_data), datasets.get_label_slice(train_data)
    
    discrimative_score_metrics(args, {'data':train_data,'label':train_label}, loaded_data)