import os
import time
import copy
import pickle
import PIL.Image
import json
import dnnlib
import psutil
import numpy as np
import torch
import torch.optim as optim

from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from training.networks import Generator, Discriminator
from wedit_network import AlphaModule
from wedit_loss import WeditGANLoss
# from metrics import metric_main
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, c, h, w = img.shape
    img = img.reshape(gh, gw, c, h, w)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * h, gw * w, c)

    assert c in [1, 3]
    if c == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if c == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def training_loop(generator: Generator,  # pretrained Generator
                  discriminator: Discriminator,  # pretrained Discriminator
                  training_set_kwargs     = {},  # Options for training set.
                  data_loader_kwargs      = {},  # Options for torch.utils.data.DataLoader.
                  g_opt_kwargs={},  # Options for generator optimizer.
                  d_opt_kwargs={},  # Options for discriminator optimizer.
                  loss_kwargs={},  # Options for loss function.
                  metrics={},  # Metrics to evaluate during training.
                  # augment_kwargs=None,          # Options for augmentation pipeline. None = disable.
                  random_seed=2022,  # Global random seed.
                  num_gpus=1,  # Number of GPUs participating in the training.
                  batch_size=4,  # Total batch size for one training iteration. Can be larger than
                                 # batch_gpu * num_gpus.
                  batch_gpu=4,  # Number of samples processed at a time by one GPU.
                  # ema_kimg=10,                  # Half-life of the exponential moving average of generator weights
                  # ema_rampup=None,              # EMA ramp-up coefficient.
                  g_reg_interval=4,  # How often to perform regularization for G?
                  # None = disable lazy regularization.
                  d_reg_interval=16,  # How often to perform regularization for D?
                  # None = disable lazy regularization.
                  rank=0,  # Rank of the current process in [0, num_gpus].
                  run_dir='.',  # Output directory.
                  # augment_p=0,                  # Initial value of augmentation probability.
                  ada_target=None,              # ADA target value. None = fixed p.
                  ada_interval=4,               # How often to perform ADA adjustment?
                  ada_kimg=500,                 # ADA adjustment speed, measured in how many kimg it takes for p to
                                                # increase/decrease by one unit.
                  total_kimg=400,  # Total length of the training, measured in thousands of real images.
                  kimg_per_tick=4,  # Progress snapshot interval.
                  image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
                  network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
                  resume_pkl=None,              # Network pickle to resume training from.
                  cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
                  allow_tf32=False,  # Enable torch.backends.cuda.matmul.allow_tf32
                                     # and torch.backends.cudnn.allow_tf32?
                  abort_fn=None,  # Callback function for determining whether to abort training.
                  # Must return consistent results across ranks.
                  # progress_fn=None,             # Callback function for updating training progress.
                  # Called for all ranks.
                  add_alpha_module=False,  # Sign for alpha module
                  device='cuda'  # Running device, default: cuda
                  ):
    # setup delta_ws which are the parameters to be trained.
    # training from scratch
    delta_ws = torch.zeros([1, generator.num_ws, generator.w_dim], requires_grad=True, device=device)
    # print(delta_ws.shape)
    alpha_module = None
    if add_alpha_module:
        alpha_module = AlphaModule(generator.num_ws, generator.w_dim).to(device)

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(
        **training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus,
                                                seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                             batch_size=batch_size // num_gpus,
                                                             **data_loader_kwargs))

    # Initialize.
    start_time = time.time()
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, generator.z_dim], device=device)
        c = torch.empty([batch_gpu, generator.c_dim], device=device)
        img = misc.print_module_summary(generator, [z, c])
        misc.print_module_summary(discriminator, [img, c])

    generator = generator.eval()

    # skip the augmentation
    # skip the distribution setup across GPUs

    print('Setting up training phases...')
    ddp_modules = dict()
    for name, module in [('G_mapping', generator.mapping), ('G_synthesis', generator.synthesis), ('D', discriminator)]:
        module.requires_grad_(False)
        ddp_modules[name] = module

    loss = WeditGANLoss(device, **ddp_modules, **loss_kwargs)
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', generator, g_opt_kwargs, g_reg_interval),
                                                   ('D', discriminator, d_opt_kwargs, d_reg_interval)]:
        # lazy regularization
        mb_ratio = reg_interval / (reg_interval + 1)
        opt_kwargs = dnnlib.EasyDict(opt_kwargs)
        opt_kwargs.lr = opt_kwargs.lr * mb_ratio
        opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
        if name == 'G':
            opt = optim.Adam([{'params': delta_ws}] if not add_alpha_module else [{'params': delta_ws},
                                                                                  {'params': alpha_module.parameters()}
                                                                                  ], **opt_kwargs)
        else:
            opt = optim.Adam(module.parameters(), **opt_kwargs)

        phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
        phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    print('Exporting sample images...')
    gw, gh = 6, 4
    grid_z = torch.randn([gw * gh, generator.z_dim], device=device).split(batch_gpu)
    grid_c = [None] * gw * gh
    images = torch.cat([generator(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
    save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1, 1], grid_size=[gw, gh])

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    while True:
        phase_real_img, c = next(training_set_iterator)
        phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
        # all_gen_z = torch.randn([batch_size * len(phases), generator.z_dim], device=device)

        for phase in phases:
            phase_gen_z = torch.randn([batch_size, generator.z_dim], device=device)
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
                phase.opt.zero_grad(set_to_none=True)
                if 'D' in phase.name:
                    phase.module.requires_grad_(True)
                elif 'G' in phase.name:
                    delta_ws.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            # for round_idx, (real_img, gen_z, cc) in enumerate(zip(phase_real_img, phase_gen_z, c)):
            # sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
            gain = phase.interval
            loss.accumulate_gradients(phase=phase.name, real_img=phase_real_img, gen_z=phase_gen_z, gen_c=c,
                                      delta_ws=delta_ws, alpha_module=alpha_module, sync=True, gain=gain)

            # Update weights.
            if 'D' in phase.name:
                phase.module.requires_grad_(False)
            elif 'G' in phase.name:
                delta_ws.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        if cur_nimg % 1000 != 0 and not done:
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        # if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
        # images = torch.cat([generator(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        images = torch.Tensor(0, 3, 256, 256)
        for z, c in zip(grid_z, grid_c):
            ws = generator.mapping(z, c)
            tgt_ws = ws + delta_ws
            image = generator.synthesis(tgt_ws).cpu()
            images = torch.cat([images, image], dim=0)
        images = images.numpy()
        save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'),
                        drange=[-1, 1], grid_size=(gw, gh))

        # Save network snapshot.
        # if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
        if cur_nimg % 10000 == 0:
            snapshot_data = dict()
            for name, module in [('D', discriminator), ('Alpha', alpha_module)]:
                if module is not None:
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module  # conserve memory

            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}-w.pkl'), 'wb') as f:
                    dw = copy.deepcopy(delta_ws).requires_grad_(False).cpu()
                    pickle.dump(('dw', dw), f)
                    del dw
            del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
    # Done.
    if rank == 0:
        print()
        print('Exiting...')
