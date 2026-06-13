import numpy as np
import copy
import time
import random
import os
import math

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.optim.meta import MetaData
from common.optim.exp.netline_scheduler_v2qq import NetLineStepLR #The scheduler choice!!!
from common.util import get_meters, test_loop
from common.util_scripts import \
    experiment_comparison, experiments_all_comparison, timing_comparison, accuracy_comparison
from common.cnn import make_resnet9, make_resnet18v2, make_resnet34v2

from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torchvision import transforms

import logging

#Experiment choice
EXP_DATASET = 'CIFAR100' #'CIFAR10','CIFAR100','SVHN'
EXP_NET = 'RESNET18' #'RESNET9', 'RESNET18', 'RESNET34'

log_filename = "run_logs/experiments_series_{}_{}.log".format(EXP_NET, EXP_DATASET)
logging.basicConfig(filename=log_filename,
                level=logging.INFO,
                format="%(levelname)s: %(asctime)s %(message)s")

#Constants
BATCH_SIZE = 128
SAMPLE_SIZE = 1024
OUTPUT_DIM = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
meta = MetaData(batch_size = BATCH_SIZE, output_dim=OUTPUT_DIM, device=DEVICE)

EXPERIMENTS = 1
EPOCHS_PER_EXPERIMENT = 50
DATASET_PATH = "./datasets"
ALPHA_DATASET=0.625

#Values validation
if EPOCHS_PER_EXPERIMENT < 20:
    raise ValueError("Incorrect dataset setting:{}, must be >= 20".format(EPOCHS_PER_EXPERIMENT))

#Manual seed
seed_value= 641
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `pytorch` pseudo-random generator at a fixed value
dummy=torch.manual_seed(seed_value)

#TF32 is switched off
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cuda.matmul.fp32_precision = 'ieee'
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
#torch.backends.cudnn.conv.fp32_precision = 'tf32'

#DATASET
train_dataloader = None
test_dataloader = None

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(*stats,inplace=True)])
valid_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

if EXP_DATASET == 'CIFAR10':
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True, transform=train_transforms)
    test_dataset = CIFAR10(root=DATASET_PATH, train=False, download=True, transform=valid_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, num_workers=0, pin_memory=True)

elif EXP_DATASET == 'CIFAR100':
    train_dataset = CIFAR100(root=DATASET_PATH, train=True, download=True, transform=train_transforms)
    test_dataset = CIFAR100(root=DATASET_PATH, train=False, download=True, transform=valid_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, num_workers=0, pin_memory=True)

elif EXP_DATASET == 'SVHN':
    train_dataset = SVHN(root=DATASET_PATH, split='train', download=True, transform=train_transforms)
    test_dataset = SVHN(root=DATASET_PATH, split='test', download=True, transform=valid_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, num_workers=0, pin_memory=True)

else:
    raise ValueError("Incorrect dataset setting:"+EXP_DATASET)

#EXPERIMENTS
'''
0-num of experiment; 1-epoch;
2-net: 0-Adam, 1-Net-line 2step, 2-SGD
3-param: 0-test accuracy (Top-1), 1-test errror (cross-entropy), 2-time per step, 3 - qq_norm_test,
4 - pq_norm, 5 - cos_phi, 6 - eta_calculated, 7 - empty, 8 - empty, 9 - empty, 10 - empty,
11 - empty, 12 - empty, 13 - empty, 14 - empty, 15 - alpha, 16 - empty,
18 - drift, 19 - alpha_drift
'''
experimental_results = np.zeros((EXPERIMENTS, EPOCHS_PER_EXPERIMENT, 3, 20))
idx_adam, idx_nl, idx_sgd = 0, 1, 2

print("Start of a series of {} experiments".format(EXPERIMENTS))
for experiment in range(EXPERIMENTS):
    #NET
    net_base = None
    if EXP_NET == 'RESNET9':
        net_base = make_resnet9(3, OUTPUT_DIM).to(DEVICE)
    elif EXP_NET == 'RESNET18':
        net_base = make_resnet18v2(3, OUTPUT_DIM).to(DEVICE)
    elif EXP_NET == 'RESNET34':
        net_base = make_resnet34v2(3, OUTPUT_DIM).to(DEVICE)
    else:
        raise ValueError("Incorrect net setting:"+EXP_NET)
    net_adam, net_snl, net_sgd = net_base, copy.deepcopy(net_base), copy.deepcopy(net_base)

    #Adam
    loss_adam = nn.CrossEntropyLoss()
    opt_adam = torch.optim.AdamW(net_adam.parameters(), lr=0.001, weight_decay=0.1)
    schd_adam = torch.optim.lr_scheduler.MultiStepLR(opt_adam, milestones=[EPOCHS_PER_EXPERIMENT*0.4, EPOCHS_PER_EXPERIMENT*0.8])

    #Net-line 2step
    snl_momentum = 0.9
    snl_eta1 = snl_eta1_min = 0.00001
    snl_foreach = True

    snl_opt = optim.SGD(net_snl.parameters(), weight_decay=5e-3, lr=snl_eta1, momentum=snl_momentum)
    snl_sch = NetLineStepLR(net_snl, snl_opt, meta, foreach=snl_foreach)
    snl_sch.do_shorten_lr_for_momentum = True
    snl_sch.do_calc_grad_norm2 = False
    snl_sch.lr_drift_size = 500
    snl_sch.init_regression()
    snl_sch.lr_averaging_check = 0.125
    snl_sch.init_averaging()

    #Sgd scheduled lr
    loss_sgd = nn.CrossEntropyLoss()
    opt_sgd = optim.SGD(net_sgd.parameters(), 0.02, momentum=0.9, weight_decay=5e-3)
    schd_sgd = optim.lr_scheduler.CosineAnnealingLR(opt_sgd, T_max=EPOCHS_PER_EXPERIMENT)

    for epoch in range(EPOCHS_PER_EXPERIMENT):
        mt_alpha, mt_adam, mt_netline, mt_sgd, mt_qq, mt_pq, mt_cos, mt_eta, mt_alpha_drift, mt_drift = get_meters(10)

        #Alpha_epoch for net-line
        snl_sch.alpha_epoch = ALPHA_DATASET

        #Drift_check for net-line
        snl_sch.lr_drift_check = epoch < EPOCHS_PER_EXPERIMENT-10
        if epoch < EPOCHS_PER_EXPERIMENT-15:
            snl_sch.lr_drift_pos = 5.0
            snl_sch.lr_drift_neg = 2.0
        elif epoch < EPOCHS_PER_EXPERIMENT-10:
            epoch_delta = EPOCHS_PER_EXPERIMENT-10-epoch
            snl_sch.lr_drift_pos = epoch_delta
            snl_sch.lr_drift_neg = 2*epoch_delta/5
        else:
            snl_sch.lr_drift_pos = 0.0
            snl_sch.lr_drift_neg = 0.0

        #Direct_mode for net-line
        direct_mode = not (epoch < EPOCHS_PER_EXPERIMENT-10)
        if direct_mode:
            eta_base1 = experimental_results[experiment, EPOCHS_PER_EXPERIMENT-12, 1, 6]
            eta_base0 = experimental_results[experiment, EPOCHS_PER_EXPERIMENT-11, 1, 6]
            eta_base = 2*eta_base0 - eta_base1
            snl_eta1 = eta_base*(1 + math.cos(math.pi*(1/2 + (epoch-(EPOCHS_PER_EXPERIMENT-10))/20)))
        else:
            snl_eta1 = snl_eta1_min
        snl_opt.param_groups[0]['lr'] = snl_eta1

        net_adam.train(True)
        net_snl.train(True)
        net_sgd.train(True)

        for train_batch in train_dataloader:
            images, labels = train_batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            #Adam
            logging.info("####Step with Adam, eta={}".format(opt_adam.param_groups[0]['lr']))
            ns_start = time.time_ns()
            logits0 = net_adam.forward(images)
            loss_val0 = loss_adam(logits0, labels)
            opt_adam.zero_grad()
            loss_val0.backward()
            opt_adam.step()
            ns_end = time.time_ns()
            mt_adam.update(ns_end-ns_start)

            #Net-line 2step
            logging.info("####Step with net-line 2step")
            ns_start = time.time_ns()
            step_result = snl_sch.step(labels, images, not direct_mode)
            ns_end = time.time_ns()
            mt_netline.update(ns_end-ns_start)
            mt_pq.update(step_result.pq_norm)
            mt_qq.update(step_result.qq_norm)
            mt_cos.update(step_result.cos_phi)
            mt_eta.update(step_result.eta)
            mt_alpha.update(step_result.alpha)

            #Sgd scheduled lr
            logging.info("####Step with SGD, eta={}".format(opt_sgd.param_groups[0]['lr']))
            ns_start = time.time_ns()
            logits2 = net_sgd.forward(images)
            loss_val2 = loss_sgd(logits2, labels)
            opt_sgd.zero_grad()
            loss_val2.backward()
            opt_sgd.step()
            ns_end = time.time_ns()
            mt_sgd.update(ns_end-ns_start)

        #Net-line statistics
        experimental_results[experiment, epoch, idx_nl, 3] = mt_qq.avg/snl_eta1
        experimental_results[experiment, epoch, idx_nl, 4] = mt_pq.avg
        experimental_results[experiment, epoch, idx_nl, 5] = mt_cos.avg
        experimental_results[experiment, epoch, idx_nl, 6] = mt_eta.avg
        experimental_results[experiment, epoch, idx_nl, 15] = mt_alpha.avg

        #timing
        experimental_results[experiment, epoch, idx_adam, 2] = mt_adam.avg
        experimental_results[experiment, epoch, idx_nl, 2] = mt_netline.avg
        experimental_results[experiment, epoch, idx_sgd, 2] = mt_sgd.avg

        # testing loop 0-Adam, 1-Net-line 2step, 2-SGD
        acc0, loss0 = test_loop(net_adam, test_dataloader, DEVICE)
        experimental_results[experiment, epoch, idx_adam, 0] = acc0
        experimental_results[experiment, epoch, idx_adam, 1] = loss0

        acc1, loss1 = test_loop(net_snl, test_dataloader, DEVICE)
        experimental_results[experiment, epoch, idx_nl, 0] = acc1
        experimental_results[experiment, epoch, idx_nl, 1] = loss1

        acc2, loss2 = test_loop(net_sgd, test_dataloader, DEVICE)
        experimental_results[experiment, epoch, idx_sgd, 0] = acc2
        experimental_results[experiment, epoch, idx_sgd, 1] = loss2
        experimental_results[experiment, epoch, idx_sgd, 6] = schd_sgd.get_last_lr()[0]

        schd_sgd.step()
        schd_adam.step()

        title="Error and loss, experiment {}".format(experiment)
        experiment_comparison(experimental_results, experiment, epoch, title=title)

        print("Experiment={}, epoch={}, sgd={}/{}, net-line={}/{}, adam={}/{}, alpha={}"\
            .format(experiment, epoch, acc2, max(experimental_results[experiment, 0:epoch+1, idx_sgd, 0]), \
                    acc1, max(experimental_results[experiment, 0:epoch+1, idx_nl, 0]), \
                    acc0, max(experimental_results[experiment, 0:epoch+1, idx_adam, 0]) , mt_alpha.avg))

experimental_mean = np.mean(experimental_results, axis=0)
experimental_var = np.var(experimental_results, axis=0)

title1="Validation on {} dataset in {} experiments".format(EXP_DATASET, EXPERIMENTS)
experiments_all_comparison(experimental_mean, experimental_var, title=title1)

title2="Time per step averaged by epochs in {} experiments".format(EXPERIMENTS)
timing_comparison(experimental_mean, title=title2)

title3 = "Error and loss, mean +/- var in {} experiments".format(EXPERIMENTS)
accuracy_comparison(experimental_results, EPOCHS_PER_EXPERIMENT, title=title3)

np.save('run_data/experimental_results.npy', experimental_results)
