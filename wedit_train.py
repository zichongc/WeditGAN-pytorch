import os
import PIL.Image
import argparse
import torch
import torchvision
import dnnlib
import pickle
import numpy as np

from training.networks import Generator, Discriminator
from torch_utils import training_stats, custom_ops
from metrics import metric_main
import wedit_training_loop


def subprocess_fn(rank, args, temp_dir=None):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    wedit_training_loop.training_loop(rank=rank, **args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='./checkpoints/ffhq70k-paper256-ada-bcr.pkl')
    parser.add_argument('--data', type=str,
                        default='./datasets/pixar.zip'
                        )
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/PX'
                        )
    parser.add_argument('--cond', type=bool, default=False)
    parser.add_argument('--mirror', type=bool, default=False)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--metrics', type=str, default='fid50k_full')

    arguments = parser.parse_args()
    gpus = arguments.gpus
    resolution = arguments.resolution
    device = torch.device('cuda')

    if not os.path.exists(arguments.output_dir):
        os.mkdir(arguments.output_dir)

    # define args for generator and discriminator
    # configurations copied from StyleGAN-ada/train.py and has been slightly modified
    with open(arguments.network, 'rb') as f:
        networks = pickle.load(f)
        G = networks['G_ema'].cuda()
        D = networks['D'].cuda()
    cfg = 'auto'
    cfg_specs = {'auto': dict(ref_gpus=-1, kimg=30, mb=-1, mbstd=-1, fmaps=-1, lrate=-1, gamma=-1, ema=-1, ramp=0.05,
                              map=8)}  # Populated dynamically based on resolution and GPU count.}
    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        spec.ref_gpus = gpus
        res = resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus)  # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus,
                         4)  # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        # spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.lrate = 0.0002
        spec.gamma = 0.0002 * (res ** 2) / spec.mb  # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args = dnnlib.EasyDict()
    args.G_kwargs = dnnlib.EasyDict(z_dim=512, w_dim=512,
                                    mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(block_kwargs=dnnlib.EasyDict(),
                                    mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4  # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256  # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    args.run_dir = arguments.output_dir
    args.total_kimg = spec.kimg
    args.num_gpus = gpus

    # Print options.
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print()

    # load pre-trained weights from pkl and copy to Generator and Discriminator
    # instead of loading pre-trained networks following guidance in StyleGAN-ada/readme.md,
    # we can diy the generator and discriminator as we want.
    print(f'initialize G and D, copy parameters from {arguments.network} ...')
    generator = Generator(c_dim=0, img_resolution=resolution, img_channels=3, **args.G_kwargs).to(device)
    pretrained_weights_dict = G.state_dict()
    model_dict = generator.state_dict()
    for k, v in model_dict.items():
        if v.size() == pretrained_weights_dict[k].size():
            model_dict[k] = pretrained_weights_dict[k]
    generator.load_state_dict(model_dict)
    discriminator = Discriminator(c_dim=0, img_resolution=resolution, img_channels=3, **args.D_kwargs).to(device)
    pretrained_weights_dict = D.state_dict()
    model_dict = discriminator.state_dict()
    for k, v in model_dict.items():
        if v.size() == pretrained_weights_dict[k].size():
            model_dict[k] = pretrained_weights_dict[k]
    discriminator.load_state_dict(model_dict)

    # configurations for training
    args.generator = generator
    args.discriminator = discriminator
    args.g_opt_kwargs = dnnlib.EasyDict(lr=spec.lrate, betas=[0, 0.99], eps=1e-8)
    args.d_opt_kwargs = dnnlib.EasyDict(lr=spec.lrate, betas=[0, 0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(r1_gamma=spec.gamma)
    args.add_alpha_module = True
    metrics = arguments.metrics.split(',')
    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise Exception(
            '\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics
    args.pop('G_kwargs')
    args.pop('D_kwargs')

    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=arguments.data,
                                               use_labels=False, max_size=None, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)  # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution  # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels  # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set)  # be explicit about dataset size

        del training_set  # conserve memory
    except IOError as err:
        raise Exception(f'--data: {err}')
    if arguments.mirror:
        args.training_set_kwargs.xflip = True

    aug = 'ada'
    args.ada_target = .6
    args.batch_size = arguments.batch

    z = torch.from_numpy(np.random.randn(arguments.batch, G.z_dim)).to(device)
    w = generator.mapping(z, None)

    assert args.num_gpus == 1, 'this implementation only supports one gpu for training.'
    subprocess_fn(rank=0, args=args)



