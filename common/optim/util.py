import torch

import math

DATASET_PATH = "./datasets"

class AverageMeter:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0
        self.val_min = float('inf')
        self.val_max = float('-inf')

    def update(self, val):
        self.count += 1
        self.val = val
        self.sum += val
        self.sum2 += val**2
        if val > self.val_max:
            self.val_max = val
        if val < self.val_min:
            self.val_min = val

    def update_values(self, val, n):
        self.count += n
        self.val = val
        self.sum += val*n

    def update_sum(self, sum_delta, n=1):
        self.count += n
        self.val = 0.0
        self.sum += sum_delta

    def avg(self):
        return self.sum / self.count

    def avg2(self):
        return self.sum2 / self.count

    def sdev_biased(self):
        return math.sqrt(max(self.avg2() - self.avg()**2, 0.0))

    def min(self):
        return self.val_min

    def max(self):
        return self.val_max

def get_meters(num):
    return [AverageMeter() for _ in range(num)]

class AverageCyclicQueue:
    def __init__(self, queue_size, fill_value, device):
        self.queue_size = queue_size
        self.fill_value = fill_value
        self.device = device
        self.reset()

    def reset(self):
        self._pos = 0
        self._pos_cyclic = False
        self._queue = \
            torch.full((self.queue_size,), fill_value=self.fill_value, dtype=torch.float).to(self.device)

    def put_value(self, value):
        self._queue[self._pos] = value
        #pos shift
        self._pos += 1
        if self._pos >= self.queue_size:
            self._pos_cyclic = True
            self._pos = 0

    def get_avg(self):
        return torch.mean(self._queue)

def cosine_annealing2_lr(eta0, eta1, epoch_cos_start, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta1
    return eta1 + 0.5*(eta0-eta1)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(epoch_cos_finish-epoch_cos_start)))