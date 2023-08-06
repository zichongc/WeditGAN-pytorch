import os
import argparse
import torch
import torchvision.utils
import tqdm
import dnnlib
import pickle
import numpy as np
from training.networks import Generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='./checkpoints/ffhq70k-paper256-ada-bcr.pkl')
    parser.add_argument('--ft_network', type=str, default=f'./outputs/AM/network-snapshot-000030.pkl')
    parser.add_argument('--dw', type=str, default=f'./outputs/AM/network-snapshot-000030-w.pkl')
    parser.add_argument('--output', type=str, default=f'./outputs/AM')
    parser.add_argument('--use_alpha', type=bool, default=True)
    parser.add_argument('--num', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--output_src', type=bool, default=False)

    arguments = parser.parse_args()
    gpus = arguments.gpus
    resolution = arguments.resolution
    seed = arguments.seed
    arguments.output = os.path.join(arguments.output, f'seed{seed}.png')
    device = torch.device('cuda')
    with open(arguments.network, 'rb') as f:
        networks = pickle.load(f)
        G = networks['G_ema'].cuda()
        D = networks['D'].cuda()

    Alpha = None
    if arguments.use_alpha:
        with open(arguments.ft_network, 'rb') as f:
            networks = pickle.load(f)
            if 'Alpha' not in networks.keys():
                raise KeyError
            Alpha = networks['Alpha'].cuda()

    # define args for generator and discriminator
    # configurations copied from StyleGAN-ada/train.py and has been slightly modified
    args = dnnlib.EasyDict()
    cfg = 'auto'
    cfg_specs = {'auto': dict(ref_gpus=-1, kimg=25000, mb=-1, mbstd=-1, fmaps=-1, lrate=-1, gamma=-1, ema=-1, ramp=0.05,
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
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb  # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(z_dim=512, w_dim=512,
                                    mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(block_kwargs=dnnlib.EasyDict(),
                                    mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    # load pre-trained weights from pkl and copy to Generator and Discriminator
    # instead of loading pre-trained networks following guidance in StyleGAN-ada/readme.md,
    print(f'initializing G and D, copying from pre-trained {arguments.network}...')
    generator = Generator(c_dim=0, img_resolution=arguments.resolution, img_channels=3, **args.G_kwargs).to(device)
    pretrained_weights_dict = G.state_dict()
    model_dict = generator.state_dict()
    for k, v in model_dict.items():
        if v.size() == pretrained_weights_dict[k].size():
            model_dict[k] = pretrained_weights_dict[k]
    generator.load_state_dict(model_dict)

    all_z = torch.from_numpy(np.random.RandomState(seed=seed).randn(arguments.num, G.z_dim)).to(device)
    with open(arguments.dw, 'rb') as f:
        dw = pickle.load(f)[1]
    delta_ws = dw.to(device)

    results = []
    for idx in tqdm.tqdm(range(arguments.num)):
        z = all_z[idx:idx+1]
        ws = generator.mapping(z, None)  # batch, n, w_dim

        if not arguments.use_alpha:
            tgt_ws = ws + delta_ws  # batch, n, w_dim    (delta_ws: 1, n, w_dim)
        else:
            batch, _ = z.shape
            _, num_ws, w_dim = delta_ws.shape
            alphas = Alpha(ws)
            tgt_ws_temp = []
            for i in range(batch):
                temp = ((1. + alphas[i]) * delta_ws[0].T).T.unsqueeze(0)
                tgt_ws_temp.append(temp)
            tgt_ws = ws + torch.cat(tgt_ws_temp, dim=0)

        tgt_img = generator.synthesis(tgt_ws)
        results.append(tgt_img)

    results = torch.cat(results, dim=0)
    torchvision.utils.save_image(results, arguments.output, nrow=arguments.num//4, normalize=True, value_range=(-1, 1))
