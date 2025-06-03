#!/bin/bash
epoch=70
lr=2e-3
dataset_=('FD001' 'FD002' 'FD003' 'FD004')
window_size_=(24 48 96)
state='all' # train,sample,eval
model_name_=('DiffUnet_wosyn')
T=1000

for model_name in "${model_name_[@]}"
    do
    for dataset in "${dataset_[@]}"
    do
        for window_size in "${window_size_[@]}"
        do
            CUDA_VISIBLE_DEVICES=3 python MainCondition.py --lr=${lr} \
            --epoch=${epoch}   --model_name=${model_name}  --dataset=${dataset}\
            --window_size=${window_size}  --state=${state}  --T=${T} 
        done
    done
done
