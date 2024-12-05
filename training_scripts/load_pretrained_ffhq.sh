cd ..
python train.py --dataset FFHQ --parallel --shuffle  --which_best FID --batch_size 20 --num_G_accumulations 1 --num_D_accumulations 1 --num_D_steps 1 --G_lr 1e-4 --D_lr 5e-4 --D_B2 0.999 --G_B2 0.999 --G_attn 0 --D_attn 0 --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 --G_ortho 0.0 --seed 99 --G_init ortho --D_init ortho --G_eval_mode --G_ch 64 --D_ch 64 --hier --dim_z 128 --ema --use_ema --ema_start 21000 --accumulate_stats --num_standing_accumulations 100  --test_every 50 --save_every 10 --num_best_copies 2 --num_save_copies 1 --seed 0 --sample_every 100  --id ffhq_unet_bce_noatt_cutmix_consist --gpus "0,1" --unconditional --unet_mixup --slow_mixup --full_batch_mixup --consistency_loss_and_augmentation --warmup_epochs 100 --base_root output --data_folder dataset --resume --resume_from pretrained_model --epoch_id ep_82
#python train.py \
#--dataset FFHQ --parallel --shuffle   \
#--which_best FID \
#--batch_size 20 --num_G_accumulations 1 --num_D_accumulations 1 \
#--num_D_steps 1 --G_lr 1e-4 --D_lr 5e-4 --D_B2 0.999 --G_B2 0.999 \
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
#--id ffhq_unet_bce_noatt_cutmix_consist --gpus "0,1" --unconditional \
#--unet_mixup --slow_mixup --full_batch_mixup --consistency_loss_and_augmentation --warmup_epochs 100 \
#--base_root output \
#--data_folder dataset \
#--resume --resume_from pretrained_model --epoch_id ep_82
config['dataset']='FFHQ'
config['parallel']=True
config['shuffle']=True
congfig['which_best']='FID'
config['batch_size']=20
config['num_G_accumulations']=1
config['num_D_accumulations']=1
config['num_D_steps']=1
config['G_lr']=1e-4
config['D_lr']=5e-4
config['D_B2']=0.999
config['G_B2']=0.999
config['G_attn']='0'
config['D_attn ']='0'
config['SN_eps']=1e-6
config['BN_eps']=1e-5
config['adam_eps']=1e-6
config['G_ortho']=0.0
config['seed']=99
config['G_init']='ortho'
config['D_init']='ortho'
config['G_eval_mode']=True
config['G_ch']=64
config['D_ch']=64
config['hier']=True
config['dim_z']=128
config['ema']=True
config['use_ema']=True
config['ema_start']=21000
config['accumulate_stats']=True
config['num_standing_accumulations']=100
config['test_every']=10000
config['save_every']=10000
config['num_best_copies']=2
config['num_save_copies']=1
config['seed']=0
config['sample_every']=4000
config['id']='ffhq_unet_bce_noatt_cutmix_consist'
config['gpus']="0,1"
config['unconditional']=True
config['unet_mixup']=True
config['slow_mixup']=True #使用warmup 计算CutMix损失

 #如果为 True，则在每个训练步骤中都扔一枚硬币。以一定的概率混合整个批次，并且 CutMix 增强损失和一致性损失是为该批次计算的唯一损失。
 # 在指定的预热时期，概率从 0 增加到 0.5。如果为 False，则为每个批次计算 CutMix 增强和一致性损失，并将其添加到默认的 GAN 损失中。
 # 在热身的情况下，增强损失乘以在指定热身时期的过程中从 0 增加到 1 的因子。
config['full_batch_mixup']=True

config['consistency_loss_and_augmentation']=True #计算 CutMix 一致性损失和CutMix 增强损失
config['warmup_epochs']=100 #在模型训练之初选用较小的学习率，X epochs后使用预设的学习率进行训练

config['resume']=True
config['resume_from']='pretrained_model'
config['epoch_id']='ep_82'