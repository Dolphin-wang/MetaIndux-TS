epoch=50
lr=2e-3
model_name_=('SSSD' 'tabddpm')
dataset_=('FD001' 'FD002' 'FD003' 'FD004')
window_size_=(24 48 96)
state='all' # train,sample,eval
schedule_name='linear' #linear, cosine
loss_type='mse' #mse mse+mmd
T=500
sample_type='ddim'

for model_name in "${model_name_[@]}"
do
    for dataset in "${dataset_[@]}"
    do
        for window_size in "${window_size_[@]}"
        do
        CUDA_VISIBLE_DEVICES=1 python MainCondition.py --lr=${lr} \
        --epoch=${epoch} --dataset=${dataset}  --model_name=${model_name}\
        --window_size=${window_size}  --state=${state}  \
        --schedule_name=${schedule_name} --loss_type=${loss_type} \
        --T=${T} --sample_type=${sample_type}
    done
done
done
