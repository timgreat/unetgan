import torch
import torchvision
import random
import utils
from utils import join_strings
import torch.nn as nn
import os

def getFFHQ():
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    config['dataset']='FFHQ'
    config['parallel']=True
    config['shuffle']=True
    config['batch_size']=1
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
    config['seed']=random.randint(0,10000)
    config['sample_every']=4000
    config['id']='ffhq_unet_bce_noatt_cutmix_consist'
    config['gpus']="0,1"
    config['unconditional']=True
    config['unet_mixup']=True
    config['slow_mixup']=True
    config['full_batch_mixup']=True
    config['consistency_loss_and_augmentation']=True
    config['warmup_epochs']=100
    config['experiment_name']='FFHQ'

    config['resume']=True
    config['resume_from']='pretrained_model'
    config['epoch_id']='ep_82'

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]

    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    device = 'cuda'
    utils.seed_rng(config['seed'])
    utils.prepare_root(config)

    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']else utils.name_from_config(config))
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
    name_suffix=config['load_weights']
    strict=True
    load_optim=True
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


    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,config['n_classes'], device=device,fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()

    which_G = G_ema if config['ema'] and config['use_ema'] else G
    with torch.no_grad():
        fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))

    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/%d.jpg'%(config['samples_root'], experiment_name,config['seed'])
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    return image_filename
