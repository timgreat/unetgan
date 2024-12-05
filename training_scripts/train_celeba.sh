python train.py --dataset celeba128 --parallel --shuffle  --which_best FID --batch_size 1 --num_G_accumulations 1 --num_D_accumulations 1 --num_D_steps 1 --G_lr 5e-5 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 --G_attn 0 --D_attn 0 --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 --G_ortho 0.0 --seed 99 --G_init ortho --D_init ortho --G_eval_mode --G_ch 64 --D_ch 64 --hier --dim_z 128 --ema --use_ema --ema_start 21000 --accumulate_stats --num_standing_accumulations 100  --test_every 10 --save_every 1 --num_best_copies 2 --num_save_copies 1 --seed 0 --sample_every 10  --id celeba128_unet_bce_noatt_cutmix_consist --gpus "0,1"  --unconditional --warmup_epochs 20 --unet_mixup --consistency_loss_and_augmentation --base_root output --data_folder dataset/images_celebA
#--dataset celeba128 --parallel --shuffle   \
#--which_best FID \
#--batch_size 50 --num_G_accumulations 1 --num_D_accumulations 1 \
#--num_D_steps 1 --G_lr 5e-5 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 \
#--G_attn 0 --D_attn 0 \
#--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
#--G_ortho 0.0 \
#--seed 99 \
#--G_init ortho --D_init ortho \
#--G_eval_mode \
#--G_ch 64 --D_ch 64 \
#--hier --dim_z 128 \
#--ema --use_ema --ema_start 21000 \
#--accumulate_stats --num_standing_accumulations 100  \
#--test_every 10000 --save_every 10000 --num_best_copies 2 --num_save_copies 1 --seed 0 \
#--sample_every 4000   \
#--id celeba128_unet_bce_noatt_cutmix_consist --gpus "0,1"  \
#--unconditional --warmup_epochs 20 \
#--unet_mixup --consistency_loss_and_augmentation \
#--base_root path/to/folder_for_results \
#--data_folder /path/to/img_align_celeba_png
config['dataset']='celeba128'#所采用的数据集
config['parallel']=True #采用多GPU
config['shuffle']=True #将训练模型的数据集进行打乱的操作
congfig['which_best']='FID' #根据哪一个指标来选择最好的checkpoint
config['batch_size']=2
config['num_G_accumulations']=1 #变相的增加batch size
config['num_D_accumulations']=1 #变相的增加batch size
config['num_D_steps']=1 #G每跑一次D跑几次
config['G_lr']=5e-5
config['D_lr']=2e-4
config['D_B2']=0.999 #adam的参数
config['G_B2']=0.999 #adam的参数
config['G_attn']='0' #attention的位置0*0
config['D_attn ']='0' #attention的位置
config['SN_eps']=1e-6 #用于光谱范数的 epsilon 值 防止normalization过程中分母出现为0的情况
config['BN_eps']=1e-5 #用于 BatchNorm 的 epsilon 值 防止normalization过程中分母出现为0的情况
config['adam_eps']=1e-6
config['G_ortho']=0.0 #正交回归参数
config['seed']=99
config['G_init']='ortho' #对G的初始化方式
config['D_init']='ortho' #对D的初始化方式
config['G_eval_mode']=True #G的评估模式
config['G_ch']=64 #G的通道数的base
config['D_ch']=64 #D的通道数的base
config['hier']=True #使用层次聚类算法
config['dim_z']=128 #噪声维度
config['ema']=True #使得G的权重更新与历史数据有关
config['use_ema']=True #使用 G 的 EMA 参数进行评估
config['ema_start']=21000 #何时开始更新EMA权重
config['accumulate_stats']=True #累加“standing”batch norm数据
config['num_standing_accumulations']=100 #在累积常规统计数据所用的前向传递次数
config['test_every']=10000 #每X次迭代测试一次
config['save_every']=10000 #每X次迭代保存一次
config['num_best_copies']=1 #要保存多少个以前的最佳检查点
config['num_save_copies']=0 #要保存多少份
config['seed']=0
config['sample_every']=50 #
config['id']='celeba128_unet_bce_noatt_cutmix_consist'
config['gpus']="0,1"
config['unconditional']=True #采用非监督学习
config['unet_mixup']=True #使用CutMix数据增强
config['consistency_loss_and_augmentation']=True #计算 CutMix 增强和一致性损失

config['resume']=True
config['resume_from']='pretrained_model'
config['epoch_id']='ep_82'