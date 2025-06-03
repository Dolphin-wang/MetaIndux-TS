#!/bin/bash
epoch=70
lr=2e-3
dataset_=('FD001' 'FD002' 'FD003' 'FD004')
window_size_=(48)
state='all' # train,sample,eval
model_name_=('SSSD')
task_=('zeroshot' 'fewshot' 'all')
T=1000

for model_name in "${model_name_[@]}"
    do
    for dataset in "${dataset_[@]}"
    do
        for window_size in "${window_size_[@]}"
        do
            for task in "${task_[@]}"
            do
                CUDA_VISIBLE_DEVICES=2 python main_fewshot.py --lr=${lr} \
                --epoch=${epoch}   --model_name=${model_name}  --dataset=${dataset}\
                --window_size=${window_size}  --state=${state}  --T=${T}  --task=${task}
                done
        done
    done
done
