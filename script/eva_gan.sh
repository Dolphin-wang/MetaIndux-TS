epoch=70
lr=2.5e-3
model_name=('TTSGAN')
dataset=('FD001')
window_size=(24 48 96)
state='eval' # train,sample,eval

for model_name in "${model_name[@]}"
do
    for dataset in "${dataset[@]}"
    do
        for window_size in "${window_size[@]}"
        do
            CUDA_VISIBLE_DEVICES=0 python MainCondition.py  --lr=${lr} \
            --epoch=${epoch}   --dataset=${dataset}  --model_name=${model_name}\
            --window_size=${window_size}  --state=${state}  
        done
    done
done
