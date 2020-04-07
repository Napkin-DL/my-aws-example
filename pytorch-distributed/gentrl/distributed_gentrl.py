import torch
import torch.nn as nn
import torch.optim as optim
from math import pi, log
from gentrl.lp import LP
import pickle

from moses.metrics.utils import get_mol

############### Additional Packages ################
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import gentrl
import logging
import os
import sys
import copy
import time
# import nvidia_smi
import GPUtil as GPU
import horovod.torch as hvd

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# 1.Model


def Net(args):

    latent_descr = 50 * [('c', 20)]
    feature_descr = [('c', 20)]
    latent_size = 50
    latent_input_size = 50

    enc = gentrl.RNNEncoder(latent_size)
    dec = gentrl.DilConvDecoder(latent_input_size, args)
    model = gentrl.DIS_GENTRL(enc, dec, latent_descr,
                              feature_descr, beta=0.001)
    return model

# 2. D-Gradients


def average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

# 3. Data Loader


def _get_data_loader(batch_size, data_dir, is_distributed):
    logger.info("Get train data loader")

    dataset = gentrl.MolecularDataset(sources=[
        {'path': data_dir,
         'smiles': 'SMILES',
         'prob': 1,
         'plogP': 'plogP',
         }],
        props=['plogP'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=1, drop_last=True, sampler=train_sampler)


class TrainStats():
    def __init__(self):
        self.stats = dict()

    def update(self, delta):
        for key in delta.keys():
            if key in self.stats.keys():
                self.stats[key].append(delta[key])
            else:
                self.stats[key] = [delta[key]]

    def reset(self):
        for key in self.stats.keys():
            self.stats[key] = []

    def print(self):
        for key in self.stats.keys():
            print(str(key) + ": {:4.4};".format(
                sum(self.stats[key]) / len(self.stats[key])
            ), end='')

        print()


class DIS_GENTRL(nn.Module):
    '''
    GENTRL model
    '''

    def __init__(self, enc, dec, latent_descr, feature_descr, tt_int=40,
                 tt_type='usual', beta=0.01, gamma=0.1):
        super(DIS_GENTRL, self).__init__()

        self.enc = enc
        self.dec = dec

        self.num_latent = len(latent_descr)
        self.num_features = len(feature_descr)

        self.latent_descr = latent_descr
        self.feature_descr = feature_descr

        self.tt_int = tt_int
        self.tt_type = tt_type

        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type)

        self.beta = beta
        self.gamma = gamma

    def get_elbo(self, x, y, host_rank):
        means, log_stds = torch.split(self.enc.encode(x).cuda(non_blocking=True),
                                      len(self.latent_descr), dim=1)
        latvar_samples = (means + torch.randn_like(log_stds) *
                          torch.exp(0.5 * log_stds))

        rec_part = self.dec.weighted_forward(x, latvar_samples).mean()

        normal_distr_hentropies = (log(2 * pi) + 1 + log_stds).sum(dim=1)

        latent_dim = len(self.latent_descr)
        condition_dim = len(self.feature_descr)

        zy = torch.cat([latvar_samples, y], dim=1)

        # GPU measure point
        gpu_perform = self.nvidia_measure(host_rank)

        log_p_zy = self.lp.log_prob(zy)

        y_to_marg = latent_dim * [True] + condition_dim * [False]
        log_p_y = self.lp.log_prob(zy, marg=y_to_marg)

        z_to_marg = latent_dim * [False] + condition_dim * [True]
        log_p_z = self.lp.log_prob(zy, marg=z_to_marg)
        log_p_z_by_y = log_p_zy - log_p_y
        log_p_y_by_z = log_p_zy - log_p_z

        kldiv_part = (-normal_distr_hentropies - log_p_zy).mean()

        elbo = rec_part - self.beta * kldiv_part
        elbo = elbo + self.gamma * log_p_y_by_z.mean()

        return elbo, {
            'loss': -elbo.detach().cpu().numpy(),
            'rec': rec_part.detach().cpu().numpy(),
            'kl': kldiv_part.detach().cpu().numpy(),
            'log_p_y_by_z': log_p_y_by_z.mean().detach().cpu().numpy(),
            'log_p_z_by_y': log_p_z_by_y.mean().detach().cpu().numpy()
        }, gpu_perform

    def nvidia_measure(self, host_rank):
        GPUs = GPU.getGPUs()
        if len(GPUs) > 1:
            gpu_host = int(host_rank)
            gpu = GPUs[gpu_host]
        else:
            gpu_host = int(os.environ['SM_CURRENT_HOST'].split('-')[1]) - 1
            gpu = GPUs[0]

        gpu_perform = [gpu_host, gpu.memoryFree,
                       gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal]
        return gpu_perform

    def save(self, folder_to_save='./'):
        if folder_to_save[-1] != '/':
            folder_to_save = folder_to_save + '/'
        torch.save(self.enc.state_dict(), folder_to_save + 'enc.model')
        torch.save(self.dec.state_dict(), folder_to_save + 'dec.model')
        torch.save(self.lp.state_dict(), folder_to_save + 'lp.model')

        pickle.dump(self.lp.order, open(folder_to_save + 'order.pkl', 'wb'))

    def load(self, folder_to_load='./'):
        if folder_to_load[-1] != '/':
            folder_to_load = folder_to_load + '/'

        order = pickle.load(open(folder_to_load + 'order.pkl', 'rb'))
        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type,
                     order=order)

        self.enc.load_state_dict(torch.load(folder_to_load + 'enc.model'))
        self.dec.load_state_dict(torch.load(folder_to_load + 'dec.model'))
        self.lp.load_state_dict(torch.load(folder_to_load + 'lp.model'))

    def sample(self, num_samples):
        z = self.lp.sample(num_samples, 50 * ['s'] + ['m'])
        smiles = self.dec.sample(50, z, argmax=False)

        return smiles


def train_as_vaelp(args):
    ######################  d-training start ###############################
    is_distributed = True
    multi_machine = False

    start = time.time()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        os.environ['WORLD_SIZE'] = str(args['size'])
        host_rank = args['rank']
        os.environ['RANK'] = str(host_rank)
        dp_device_ids = [host_rank]
        torch.cuda.set_device(host_rank)

    if args['hvd']:
        hvd.init()
        # # Horovod: pin GPU to local rank
        # print("hvd.local_rank() : {} ".format(hvd.local_rank()))
        # torch.cuda.set_device(host_rank)

    verbose_step = args['verbose_step']

    train_loader = _get_data_loader(
        args['batch_size'], args['data_dir'], is_distributed)

    logger.info("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    model = Net(args)
    model = model.to(device)

    if is_distributed and not args['apex']:
        if multi_machine and use_cuda:
            # multi-machine multi-gpu case
            model = torch.nn.parallel.DistributedDataParallel(model).to(device)
        else:
            # single-machine multi-gpu case or single-machine or multi-machine cpu case
            model = torch.nn.DataParallel(model).to(device)
    elif args['apex']:
        if args['sync_bn']:
            import apex
            print("using apex synced BN")
            model = apex.parallel.convert_syncbn_model(model)

        ######################  d-training end ###############################
    if args['hvd']:
        lr_scaler = hvd.size()
        optimizer = optim.Adam(model.parameters(), lr=args['lr'] * lr_scaler)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    if args['apex']:
        # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
        # for convenient interoperation with argparse.
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args['opt_level'],
                                          keep_batchnorm_fp32=args['keep_batchnorm_fp32'],
                                          loss_scale=args['loss_scale']
                                          )
    if args['apex'] and is_distributed:
        model = DDP(model, delay_allreduce=True)

    global_stats = TrainStats()
    local_stats = TrainStats()

    epoch_i = 0
    to_reinit = False
    buf = None

    ############## Best weight Store ##############
    best_elbo = 10000.0
    best_model_wts = copy.deepcopy(model.state_dict())

    while epoch_i < args['num_epochs']:
        epoch_start = time.time()
        i = 0
        t_elbo = 0.0

        if epoch_i in [0, 1, 5]:
            to_reinit = True

        epoch_i += 1

        if verbose_step:
            print("Epoch", epoch_i, ":")

        batch_cnt = 0
        for x_batch, y_batch in train_loader:
            if verbose_step:
                print("!", end='')

            i += 1
            # move labels from cpu to gpu
            y_batch = y_batch.cuda(non_blocking=True)
            y_batch = y_batch.float().to(
                model.module.lp.tt_cores[0].cuda())
            if len(y_batch.shape) == 1:
                y_batch = y_batch.view(-1, 1).contiguous()

            if to_reinit:
                if (buf is None) or (buf.shape[0] < 5000):
                    enc_out = model.module.enc.encode(x_batch)
                    means, log_stds = torch.split(enc_out,
                                                  len(model.module.latent_descr),
                                                  dim=1)
                    z_batch = (means + torch.randn_like(log_stds) *
                               torch.exp(0.5 * log_stds))
                    cur_batch = torch.cat([z_batch, y_batch], dim=1)
                    if buf is None:
                        buf = cur_batch
                    else:
                        buf = torch.cat([buf, cur_batch])
                else:
                    descr = len(model.module.latent_descr) * [0]
                    descr += len(model.module.feature_descr) * [1]
                    model.module.lp.reinit_from_data(buf, descr)
                    model.module.lp.cuda()
                    buf = None
                    to_reinit = False

                continue

            elbo, cur_stats, gpu_perform = model.module.get_elbo(
                x_batch, y_batch, host_rank)

            local_stats.update(cur_stats)
            global_stats.update(cur_stats)

            optimizer.zero_grad()
            loss = -elbo

            if args['apex']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if is_distributed and not use_cuda:
                #     # average gradients manually for multi-machine cpu case only
                average_gradients(model)

            optimizer.step()

            if verbose_step and i % verbose_step == 0:
                local_stats.print()
                local_stats.reset()
                i = 0

            t_elbo += elbo.detach().cpu().numpy()
            batch_cnt += 1
        # FOR end

        cur_elbo = -t_elbo/batch_cnt

        if args['hvd']:
            cur_elbo = _metric_average(cur_elbo, 'cur_elbo')

        if cur_elbo <= best_elbo:
            best_elbo = cur_elbo
            best_model_wts = copy.deepcopy(model.state_dict())

        # epoch_i += 1
        if i > 0:
            local_stats.print()
            local_stats.reset()

        init_intval = '{:0.3f}'.format(time.time()-start)
        epoch_intval = '{:0.3f}'.format(time.time()-epoch_start)

        print("Total_time: {0}, Epoch_time: {1}, HOST_NUM : {2}, GPU RAM Free: {3:.0f}MB | Used: {4:.0f}MB | Util {5:3.0f}% | Total {6:.0f}MB | Loss : {7}".format(
            init_intval, epoch_intval, gpu_perform[0], gpu_perform[1], gpu_perform[2], gpu_perform[3], gpu_perform[4], cur_elbo))

        print('HOST_NUM: {} | Allocated: {} GB | Cached: {} GB'.format(host_rank, round(
            torch.cuda.memory_allocated(int(host_rank))/1024**3, 1), round(torch.cuda.memory_cached(int(host_rank))/1024**3, 1)))
    # FOR end

    model.load_state_dict(best_model_wts)
    model.module.save('./saved_gentrl/')

    return global_stats


def _metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train_as_rl(args):

    is_distributed = True
    multi_machine = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        os.environ['WORLD_SIZE'] = str(args['size'])
        host_rank = args['rank']
        os.environ['RANK'] = str(host_rank)
        dp_device_ids = [host_rank]
        torch.cuda.set_device(host_rank)

    model = Net(args)
    model = model.to(device)
    model.load('saved_gentrl/')

    reward_fn = args['reward_fn']
    batch_size = args['batch_size']
    verbose_step = args['verbose_step']
    num_iterations = args['num_iterations']

    if is_distributed:
        if multi_machine and use_cuda:
            # multi-machine multi-gpu case
            model = torch.nn.parallel.DistributedDataParallel(model).to(device)
        else:
            # single-machine multi-gpu case or single-machine or multi-machine cpu case
            model = torch.nn.DataParallel(model).to(device)

    optimizer_lp = optim.Adam(model.module.lp.parameters(), lr=args['lr_lp'])
    optimizer_dec = optim.Adam(
        model.module.dec.latent_fc.parameters(), lr=args['lr_dec'])

    global_stats = TrainStats()
    local_stats = TrainStats()

    cur_iteration = 0

    while cur_iteration < num_iterations:
        print("!", end='')

        exploit_size = int(batch_size * (1 - 0.3))
        exploit_z = model.module.lp.sample(exploit_size, 50 * ['s'] + ['m'])

        z_means = exploit_z.mean(dim=0)
        z_stds = exploit_z.std(dim=0)

        expl_size = int(batch_size * 0.3)
        expl_z = torch.randn(expl_size, exploit_z.shape[1])
        expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
        expl_z += z_means[None, :]

        z = torch.cat([exploit_z, expl_z])
        smiles = model.module.dec.sample(50, z, argmax=False)
        zc = torch.zeros(z.shape[0], 1).to(z.device)
        conc_zy = torch.cat([z, zc], dim=1)
        log_probs = model.module.lp.log_prob(
            conc_zy, marg=50 * [False] + [True])
        log_probs += model.module.dec.weighted_forward(smiles, z)
        r_list = [reward_fn(s) for s in smiles]

        rewards = torch.tensor(r_list).float().to(exploit_z.device)
        rewards_bl = rewards - rewards.mean()

        optimizer_dec.zero_grad()
        optimizer_lp.zero_grad()
        loss = -(log_probs * rewards_bl).mean()
        loss.backward()

        if is_distributed and not use_cuda:
            #     # average gradients manually for multi-machine cpu case only
            average_gradients(model)

        optimizer_dec.step()
        optimizer_lp.step()

        valid_sm = [s for s in smiles if get_mol(s) is not None]
        cur_stats = {'mean_reward': sum(r_list) / len(smiles),
                     'valid_perc': len(valid_sm) / len(smiles)}

        local_stats.update(cur_stats)
        global_stats.update(cur_stats)

        cur_iteration += 1

        if verbose_step and (cur_iteration + 1) % verbose_step == 0:
            local_stats.print()
            local_stats.reset()

    model.module.save('./saved_gentrl_after_rl/')
    return global_stats
