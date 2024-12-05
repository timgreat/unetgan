import random

import torch
import torchvision
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import utils
from BigGAN import Generator
from utils import join_strings
import torch.nn as nn
import os
def getCeleba():
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    config['dataset']='celeba128'#所采用的数据集
    config['parallel']=True #采用多GPU
    config['shuffle']=True #将训练模型的数据集进行打乱的操作
    config['batch_size']=4
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
    config['seed']=random.randint(0,10000)
    config['sample_every']=50 #
    config['id']='celeba128_unet_bce_noatt_cutmix_consist'
    config['gpus']="0,1"
    config['unconditional']=True #采用非监督学习
    config['unet_mixup']=True #使用CutMix数据增强
    config['consistency_loss_and_augmentation']=True #计算 CutMix 增强和一致性损失
    config['experiment_name'] = 'Celeba'

    config['resume']=True
    config['resume_from']='pretrained_model1'
    config['epoch_id']='ep_127'

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]

    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    device = 'cuda'
    utils.seed_rng(config['seed'])
    utils.prepare_root(config)

    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))
    G = model.Generator(**config).to(device)
    G_ema = model.Generator(**{**config, 'skip_init': True,
                               'no_optim': True}).to(device)
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    G_batch_size = int(G_batch_size * config["num_G_accumulations"])
    z, y = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])

    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}
    epoch_id = config["epoch_id"]

    root = config["resume_from"]
    name_suffix = config['load_weights']
    strict = True
    load_optim = True
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    print("epoch id : ", epoch_id)
    if G is not None:
        print('%s/%s.pth' % (root, join_strings('_', ['G', epoch_id, name_suffix])))
        G.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G', epoch_id, name_suffix]))),
            strict=strict)
        if load_optim:
            s = torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', epoch_id, name_suffix])))
            print(">>", len(s))
            # print(s)
            G.optim.load_state_dict(
                torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', epoch_id, name_suffix]))))

    if G_ema is not None:
        print('%s/%s.pth' % (root, join_strings('_', ['G_ema', epoch_id, name_suffix])))
        G_ema.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', epoch_id, name_suffix]))),
            strict=strict)

    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()

    which_G = G_ema if config['ema'] and config['use_ema'] else G
    with torch.no_grad():
        fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))

    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/%d.jpg' % (config['samples_root'], experiment_name, config['seed'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename, nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    return image_filename