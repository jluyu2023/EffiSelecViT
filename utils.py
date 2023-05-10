# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


def get_zeta(model):
    # 这个方法的作用就是将具有梯度的zeta转化为列表存储起来
    zeta = []
    for name, param in model.named_parameters():
        if 'head_zeta' in name:
            zeta.append(param.cpu().detach().reshape(-1).numpy().tolist())  # reshape(-1）是将一个多维矩阵改成一长行

    head_zetas = [z for k in zeta for z in k]  # 进一步展平，展平为一行
    return head_zetas


def get_mlp_zeta(model):
    # 这个方法的作用就是将具有梯度的zeta转化为列表存储起来
    zeta = []
    for name, param in model.named_parameters():
        if 'mlp_zeta' in name:
            zeta.append(param.cpu().detach().reshape(-1).numpy().tolist())  # reshape(-1）是将一个多维矩阵改成一长行

    mlp_zetas = [z for k in zeta for z in k]  # 进一步展平，展平为一行
    return mlp_zetas


def get_token_zeta(model):
    zeta = []
    for name, param in model.named_parameters():
        if 'token_zeta' in name:
            zeta.append(param.cpu().detach().reshape(-1).numpy().tolist())  # reshape(-1）是将一个多维矩阵改成一长行

    token_zetas = [z for k in zeta for z in k]  # 进一步展平，展平为一行
    return token_zetas


def get_threshold(checkpoint_path, head_prune_ratio, mlp_prune_ratio):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print('搜索文件的搜索epoch为：', checkpoint['epoch'])
    head_zeta = []
    mlp_zeta = []
    head_threshold = None
    mlp_threshold = None
    for i in checkpoint['model']:
        if 'head_zeta' in i:
            # print(checkpoint['model'][i])
            head_zeta.append(checkpoint['model'][i])
        if 'mlp_zeta' in i:
            mlp_zeta.append(checkpoint['model'][i])

    if len(head_zeta) == 0:
        print('配置文件中没有head_zeta信息')
    else:
        print('开始进行计算head_threshold，剪枝率为：', head_prune_ratio)
        head_data = []
        for i in range(len(head_zeta)):
            head_data.append(head_zeta[i].squeeze().reshape(-1).numpy().tolist())

        head_data = [z for k in head_data for z in k]  # 进一步展平，展平为一行
        head_data = sorted(head_data)
        print('head_data:', head_data)
        min_index = int(head_prune_ratio * len(head_data))
        head_threshold = head_data[min_index - 1]
        print('head_threshold=', head_threshold, type(head_threshold))

        head_cfg_mask = []
        for i in range(len(head_zeta)):
            head_cfg_mask.append(
                (head_zeta[i] > head_threshold).int()
                # 得到每层layer的0，1值
            )
        torch.cuda.empty_cache()
        head_cfg_mask_data = []
        for i in range(len(head_cfg_mask)):
            print(head_cfg_mask[i].squeeze().reshape(-1).numpy().tolist())
            head_cfg_mask_data.append(head_cfg_mask[i].squeeze().numpy().tolist())
        print(head_cfg_mask_data)

    if len(mlp_zeta) == 0:
        print('配置文件中没有mlp_zeta信息')
    else:
        print('开始进行计算mlp_threshold，剪枝率为：', mlp_prune_ratio)
        mlp_data = []
        for i in range(len(mlp_zeta)):
            mlp_data.append(mlp_zeta[i].squeeze().reshape(-1).numpy().tolist())

        mlp_data = [z for k in mlp_data for z in k]  # 进一步展平，展平为一行
        mlp_data = sorted(mlp_data)
        # print('mlp_data:', mlp_data)
        min_index = int(mlp_prune_ratio * len(mlp_data))
        mlp_threshold = mlp_data[min_index - 1]
        print('mlp_threshold=', mlp_threshold, type(mlp_threshold))

        mlp_cfg_mask = []
        for i in range(len(mlp_zeta)):
            mlp_cfg_mask.append(
                (mlp_zeta[i] > mlp_threshold).int()
                # 得到每层layer的0，1值
            )
        torch.cuda.empty_cache()
        mlp_cfg_mask_data = []
        for i in range(len(mlp_cfg_mask)):
            # print(mlp_cfg_mask[i].squeeze().reshape(-1).numpy().tolist())
            mlp_cfg_mask_data.append(mlp_cfg_mask[i].squeeze().numpy().tolist())
        # print(mlp_cfg_mask_data)

    return [head_threshold, mlp_threshold]


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema': checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
