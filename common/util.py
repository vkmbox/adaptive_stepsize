import numpy as np
from scipy.special import softmax

import logging

DATASET_PATH = "./datasets"

class AverageMeter:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(prediction, target):
    # Note that prediction.shape == target.shape == [B, ]
    matching = (prediction == target).float()
    return matching.mean().item()

def calculate_accuracy_np(prediction, target):
    # Note that prediction.shape == target.shape == [B, ]
    return np.average(prediction ==target)

def loss_crossentropy_np(zz_logits, true_labels):
    batch_size = len(true_labels)
    qq = softmax(zz_logits, axis=(0)) + 1e-8
    pp = np.zeros_like(qq)
    for col in range(batch_size):
        pp[true_labels[col],col]=1
    return -np.sum(pp * np.log(qq))/batch_size
            
def labels_to_softhot_np(true_labels, output_dim):
    batch_size = true_labels.shape[0]
    yy_softhot = np.zeros((output_dim, batch_size))
    for batch_num in range(batch_size):
        yy_softhot[true_labels[batch_num], batch_num] = 1.0
    return yy_softhot
