import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix


class Loss:
    def accumulate_gradients(self, phase, real_img, gen_z, gen_c, delta_ws, alpha_modules, sync, gain):
        raise NotImplementedError()


class WeditGANLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None,
                 style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2,
                 pl_decay=0.01, pl_weight=2, perp_gamma=0.0005, alpha_reg_gamma=0.1):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.perp_gamma = perp_gamma
        self.alpha_reg_gamma = alpha_reg_gamma

    def run_G(self, z, c, delta_ws, sync, alpha_module=None):
        alphas = None
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)       # batch, n, w_dim
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

            if alpha_module is None:
                tgt_ws = ws + delta_ws          # batch, n, w_dim    (delta_ws: 1, n, w_dim)
            else:
                # # add AlphaModules
                # tgt_ws = ws + (1+alpha)*delta_ws
                batch, _ = z.shape
                _, num_ws, w_dim = delta_ws.shape
                alphas = alpha_module(ws)
                tgt_ws_temp = []
                for i in range(batch):
                    temp = ((1.+alphas[i])*delta_ws[0].T).T.unsqueeze(0)
                    tgt_ws_temp.append(temp)
                tgt_ws = ws + torch.cat(tgt_ws_temp, dim=0)

        with misc.ddp_sync(self.G_synthesis, sync):
            src_img = self.G_synthesis(ws)
            tgt_img = self.G_synthesis(tgt_ws)

        return src_img, tgt_img, ws, tgt_ws, alphas

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, gen_z, gen_c, delta_ws, alpha_module, sync, gain,
                             do_Gperp=True, do_Galphareg=True):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Gperp', 'Galphareg', 'Gcl', 'Dmain', 'Dreg', 'Dboth', 'Dcl']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_src_img, gen_tgt_img, _gen_src_ws, _gen_tgt_ws, alphas = self.run_G(gen_z, gen_c, delta_ws,
                                                                                sync=(sync and not do_Gpl),
                                                                                alpha_module=alpha_module)
                gen_logits = self.run_D(gen_tgt_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_perp, loss_cl, loss_alpha_reg = 0., 0., 0.
                if do_Gperp:
                    # compute perpendicular loss (Eq. 5)
                    batch, num_ws, w_dim = _gen_src_ws.shape
                    for i in range(batch):
                        for j in range(batch):
                            if i == j:
                                continue
                            loss_perp += (torch.mm(delta_ws[0],
                                                   (_gen_src_ws[i]-_gen_src_ws[j]).T)*torch.eye(num_ws, device=self.device)).sum()/num_ws
                    training_stats.report('Loss/G/loss_lperp', loss_perp)

                if do_Galphareg:
                    # alpha module regularization
                    if alphas is None:
                        raise ValueError
                    loss_alpha_reg = torch.sqrt(alphas**2).mean()
                    training_stats.report('Loss/G/loss_alpha_reg', loss_perp)

                # -log(sigmoid(gen_logits))
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) + loss_perp * self.perp_gamma + loss_cl + loss_alpha_reg * self.alpha_reg_gamma
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                # gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], delta_ws, sync=sync)
                gen_src_img, gen_tgt_img, _gen_src_ws, _gen_tgt_ws, alphas = self.run_G(gen_z[:batch_size], gen_c[:batch_size],
                                                                                delta_ws, sync=sync,
                                                                                alpha_module=alpha_module)
                pl_noise = torch.randn_like(gen_tgt_img) / np.sqrt(gen_tgt_img.shape[2] * gen_tgt_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_tgt_img * pl_noise).sum()], inputs=[_gen_tgt_ws],
                                                   create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_tgt_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_src_img, gen_tgt_img, _gen_src_ws, _gen_tgt_ws, alphas = self.run_G(gen_z, gen_c, delta_ws,
                                                                                sync=(sync and not do_Gpl),
                                                                                alpha_module=alpha_module)

                gen_logits = self.run_D(gen_tgt_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            real_c = gen_c
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                # print(real_img)
                real_img_tmp = real_img[0].detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp],
                                                       create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


