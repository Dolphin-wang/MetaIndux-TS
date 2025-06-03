
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
# LSTM希望输入是三维numpy数组的形状，我需要相应地转换训练和测试数据。
def gen_train(id_df, seq_length, seq_cols):
 
    data_array = id_df[seq_cols].values
    #存储的array的shape,第一个维度必须是0，有且仅有这一个，代表这个维度是可拓展的。
    num_elements = data_array.shape[0]
    lstm_array=[]
    
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)
    

def gen_target(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length-1:num_elements+1]


def gen_test(id_df, seq_length, seq_cols, mask_value=0):
    df_mask = pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    df_mask[:] = mask_value
    
    id_df = df_mask.append(id_df,ignore_index=True)
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]

    start = num_elements-seq_length
    stop = num_elements
    
    lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)

def train_data_load(dataset='FD001',sequence_length=50):

    feats = ['Sensor2', 'Sensor3', 'Sensor4', 'Sensor7', 'Sensor8', 'Sensor9', 'Sensor11', 'Sensor12', 'Sensor13', 'Sensor14', 'Sensor15', 'Sensor17', 'Sensor20', 'Sensor21']
    df_train = pd.read_csv('dataset/train_norm_'+dataset + '.csv')
    x_train=np.concatenate(list(list(gen_train(df_train[df_train['UnitNumber']==unit], sequence_length, feats)) 
                                for unit in df_train['UnitNumber'].unique()))
    print(x_train.shape)

    y_train = np.concatenate(list(list(gen_target(df_train[df_train['UnitNumber']==unit], sequence_length, "RUL")) 
                              for unit in df_train['UnitNumber'].unique()))

    print(y_train.shape)
    return torch.tensor(x_train).float(),torch.tensor(y_train).float().unsqueeze(-1)
def test_data_load(dataset='FD001',sequence_length=50):

    feats = ['Sensor2', 'Sensor3', 'Sensor4', 'Sensor7', 'Sensor8', 'Sensor9', 'Sensor11', 'Sensor12', 'Sensor13', 'Sensor14', 'Sensor15', 'Sensor17', 'Sensor20', 'Sensor21']
    df_test = pd.read_csv('dataset/test_norm_'+ dataset + '.csv')
    y_train=np.concatenate(list(list(gen_test(df_test[df_test['UnitNumber']==unit], sequence_length, feats)) 
                            for unit in df_test['UnitNumber'].unique()))
    print(y_train.shape)

    y_true = pd.read_csv('./data/RUL_'+dataset+'.txt',delim_whitespace=True,names=["RUL"])
    y_test = y_true.RUL.values
    print(y_test.shape)
    return torch.tensor(y_train).float(),torch.tensor(y_test).float().unsqueeze(-1)
def load_RUL2012(seq_length=80, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' ) ):
    path = './PHM2012/'

    d1 = []
    # for i in range(1,8):
    for i in range(1,8):
        d1.append(pd.read_csv(path+"Data1_"+str(i)+".csv", index_col=None, header=None).iloc[:, -2])
    df1 = pd.concat(d1, axis=0)  # 合并

    data1 = df1.values

    
    seq_length = seq_length # d_model * seq_length = 2560
    d_model = 2560 // seq_length
    
    data = torch.tensor(data1).reshape(-1, seq_length, d_model)

    l = np.concatenate((np.ones(1296).reshape(-1, 1), np.linspace(1, 0, 1507).reshape(-1, 1)), axis=0)
    l = np.concatenate((l, np.ones(824).reshape(-1, 1), np.linspace(1, 0, 47).reshape(-1, 1)), axis=0)
    l = np.concatenate((l, np.ones(1350).reshape(-1, 1), np.linspace(1, 0, 1025).reshape(-1, 1)), axis=0)
    l = np.concatenate((l, np.ones(1082).reshape(-1,1), np.linspace(1, 0, 57).reshape(-1,1)), axis=0)
    l = np.concatenate((l, np.ones(2410).reshape(-1,1), np.linspace(1, 0, 53).reshape(-1,1)), axis=0)
    l = np.concatenate((l, np.ones(2402).reshape(-1,1), np.linspace(1, 0, 46).reshape(-1,1)), axis=0)
    l = np.concatenate((l, np.ones(2206).reshape(-1,1), np.linspace(1, 0, 53).reshape(-1,1)), axis=0)

    # l = np.linspace(1, 0, 2803).reshape(-1,1)
    # l = np.concatenate((l, np.linspace(1, 0, 871).reshape(-1,1)), axis=0)
    # l = np.concatenate((l, np.linspace(1, 0, 2375).reshape(-1,1)), axis=0)
    # l = np.concatenate((l, np.linspace(1, 0, 1139).reshape(-1,1)), axis=0)
    # l = np.concatenate((l, np.linspace(1, 0, 2463).reshape(-1,1)), axis=0)
    # l = np.concatenate((l, np.linspace(1, 0, 2448).reshape(-1,1)), axis=0)
    # l = np.concatenate((l, np.linspace(1, 0, 2259).reshape(-1,1)), axis=0)

    data = torch.tensor(data).float().to(device)
    label = torch.tensor(l).float().to(device)
    train_data,train_label = data[0:3674, :, :], label[0:3674]
    test_data,test_label = data[3674:-1, :, :], label[3674:-1]
    return train_data,train_label,test_data,test_label
if __name__ == "__main__":
    train_data_load()
    test_data_load()