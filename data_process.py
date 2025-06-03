import numpy as np
import torch
import os

def load_train_data_rul(sequence_length = 50, stride = 1, dataset = 's02'):
    dataset_path = 'N-CMAPSS/' + dataset + '/train'
    file_lists = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    data = np.load(file_lists[0])
    max_rul = np.full((data.shape[0],1),fill_value=data[:,0].max())/100 
    data = np.concatenate((data,max_rul),axis=1)
    for index_file in file_lists:
        index_data = np.load(index_file)
        max_rul = np.full((index_data.shape[0],1),fill_value=index_data[:,0].max())/100 
        index_data = np.concatenate((index_data,max_rul),axis=1)
        data = np.concatenate((data,index_data), axis=0)

    window_list = []
    label_list = []
    num_samples = int((data.shape[0] - sequence_length)/stride) + 1

    for i in range(num_samples):
        window = data[i*stride:i*stride + sequence_length, 4:]  # each individual window
        label = data[i*stride + sequence_length-1, 0]
        window_list.append(window)
        label_list.append([label])

    sample_array = np.array(window_list).astype(np.float32)
    
    label_array = np.array(label_list).astype(np.float32)

    data = torch.from_numpy(sample_array)
    label = torch.from_numpy(label_array)
    return data,label

def load_test_data_rul(sequence_length = 50, stride = 1, dataset='s02'):
    dataset_path = 'N-CMAPSS/' + dataset + '/test'
    file_lists = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    data = np.load(file_lists[0])
    max_rul = np.full((data.shape[0],1),fill_value=data[:,0].max())/100
    data = np.concatenate((data,max_rul),axis=1)
    for index_file in file_lists:
        index_data = np.load(index_file)
        max_rul = np.full((index_data.shape[0],1),fill_value=index_data[:,0].max())/100 
        index_data = np.concatenate((index_data,max_rul),axis=1)
        data = np.concatenate((data,index_data), axis=0)
    window_list = []
    label_list = []    
    num_samples = int((data.shape[0] - sequence_length)/stride) + 1

    for i in range(num_samples):
        window = data[i*stride:i*stride + sequence_length, 4:]  # each individual window
        label = data[i*stride + sequence_length-1, 0]
        window_list.append(window)
        label_list.append([label])

    sample_array = np.array(window_list).astype(np.float32) 
    label_array = np.array(label_list).astype(np.float32)

    data = torch.from_numpy(sample_array)
    label = torch.from_numpy(label_array)
    return data,label

def load_train_data(sequence_length = 50, stride = 1, dataset='s02'):
    dataset_path = 'N-CMAPSS/' + dataset + '/train'
    file_lists = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    data = np.load(file_lists[0])
    for index_file in file_lists:
        index_data = np.load(index_file)
        data = np.concatenate((data,index_data), axis=0)

    window_list = []
    label_list = []
    num_samples = int((data.shape[0] - sequence_length)/stride) + 1

    for i in range(num_samples):
        window = data[i*stride:i*stride + sequence_length, 4:]  # each individual window
        label = data[i*stride + sequence_length-1, 0]
        window_list.append(window)
        label_list.append([label])

    sample_array = np.array(window_list).astype(np.float32)
    
    label_array = np.array(label_list).astype(np.float32)

    data = torch.from_numpy(sample_array)
    label = torch.from_numpy(label_array)
    return data,label

def load_test_data(sequence_length = 50, stride = 1, dataset='s02'):
    dataset_path = 'N-CMAPSS/' + dataset + '/test'
    file_lists = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    data = np.load(file_lists[0])
    for index_file in file_lists:
        index_data = np.load(index_file)
        data = np.concatenate((data,index_data), axis=0)
    window_list = []
    label_list = []    
    num_samples = int((data.shape[0] - sequence_length)/stride) + 1

    for i in range(num_samples):
        window = data[i*stride:i*stride + sequence_length, 4:]  # each individual window
        label = data[i*stride + sequence_length-1, 0]
        window_list.append(window)
        label_list.append([label])

    sample_array = np.array(window_list).astype(np.float32) 
    label_array = np.array(label_list).astype(np.float32)

    data = torch.from_numpy(sample_array)
    label = torch.from_numpy(label_array)
    return data,label