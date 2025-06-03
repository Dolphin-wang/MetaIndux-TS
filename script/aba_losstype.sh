epoch=70
lr=2e-3
arch='att+td' # 'att' 'original' 'att+td' 'td'
dataset=('FD001' 'FD002' 'FD003' 'FD004')
window_size=48
state='all' # train,sample,eval
schedule_name=('linear' 'cosine') #linear, cosine
loss_type=('mse' 'mse+mmd') #mse mse+mmd
T=500
for ((i=0;i<4;i++))
do
    for ((j=0;j<2;j++))
    do
        for ((k=0;k<2;k++))
        do
        CUDA_VISIBLE_DEVICES=1 python MainCondition.py  --lr=${lr} \
        --epoch=${epoch} --arch=${arch}  --dataset=${dataset[$i]}  \
        --window_size=${window_size}  --state=${state}  \
        --schedule_name=${schedule_name[$k]} --loss_type=${loss_type[$j]} \
        --T=${T}
        done
    done
done

# lr_type="multistep"
# embedding=14
# lstm_hidden=64
# python main.py --lr_type=${lr_type} --embedding=${embedding} --lstm_hidden=${lstm_hidden}