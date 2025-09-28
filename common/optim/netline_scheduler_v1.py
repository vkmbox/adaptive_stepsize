import torch
from torch import Tensor
from torch.linalg import norm
import torch.nn.functional as F
from typing import List, Optional

import math
import logging

#force_trainmode=True/False
def snl_forward(net, images, force_trainmode):
    if force_trainmode == True:
        training = net.training
        net.train(True)
        logging.info("##Snl: --==Explicit train forward==--")
        logits = net.forward(images)
        net.train(training)
        return logits
    else:
        return net.forward(images)
    
def snl_eta(eta_test, delta_pq, delta_qq, norm_pq, norm_qq, epsilon, beta_min, do_logging):
    with torch.no_grad():
        cos_phi = torch.sum(delta_pq*delta_qq)/(norm_pq*norm_qq + epsilon)
        eta_next = norm_pq*cos_phi*eta_test/torch.maximum(norm_qq, beta_min)
        if do_logging:
            logging.info("##Snl: cos(pp^qq)={}, norm_pq={}, norm_qq={}, eta_test={}, eta_raw={}, beta_min={}"\
                        .format(cos_phi, norm_pq, norm_qq, eta_test, eta_next, beta_min))
        return eta_next, cos_phi

class StepResult:
    def __init__(self, eta, pq_norm=0.0, qq_norm=0.0, cos_phi=0.0, alpha = None, grad_norm2_squared=None, accum_norm2_squared=None):
        self.eta = eta
        self.pq_norm = pq_norm
        self.qq_norm = qq_norm
        self.cos_phi = cos_phi
        self.alpha = alpha
        self.grad_norm2_squared = grad_norm2_squared
        self.accum_norm2_squared = accum_norm2_squared

class NetLineStepLR:

    #Values for lr, momentum and weight_decay are set externally in optimiser
    def __init__(self, net, optimizer, meta): #, foreach=False)
        self.net = net
        self.optimizer = optimizer
        self.meta = meta

        self.eta1 = optimizer.param_groups[0]['lr'] #1st step eta-size
        self.alpha_epoch = 0.9 #eta multiplier
        self.beta_min = torch.tensor(0.00001).to(meta.device) #min for eta denom for the eta-calculation stability
        self.epsilon = 1e-9

        self.dropout_mode = False #Set true if the net uses dropout layers
        self.do_logging = False #Is additional params logging performed or not, the logging may affect performance
        self.do_calc_grad_norm2 = False #Is norm2 squared of gradient calculated or not, the calculation may affect performance
        self.do_shorten_lr_for_momentum = False #If momentum > 0, shorten lr by theoretical ratio |g|/|v|

    def step(self, labels, images):
        net = self.net
        meta = self.meta
        optimizer = self.optimizer

        momentum = optimizer.param_groups[0]['momentum']
        alpha_momentum = math.sqrt(1-momentum**2) if momentum > 0.0 and self.do_shorten_lr_for_momentum else 1.0

        if self.dropout_mode and net.training:
            raise ValueError("For dropout_mode == True net.training must be False")
        logging.info("##Snl: Step start, calculating pp")
        with torch.no_grad():
            pp = F.one_hot(labels, meta.output_dim).to(meta.device)
        logging.info("##Snl: calculating logits and qq0")
        net.zero_grad()
        snl_logits = snl_forward(net, images, self.dropout_mode) ## new gradient with dropout is generated here (1*)
        with torch.no_grad():
            snl_logits0 = snl_logits if self.dropout_mode == False else snl_forward(net, images, False)
            qq0 = F.softmax(snl_logits0, dim=1) + self.epsilon ## all qqxx calculated with dropout off
        logging.info("##Snl: calculating criterion")
        snl_loss = F.cross_entropy(snl_logits, labels, reduction='mean')
        logging.info("##Snl: performing small step")
        snl_loss.backward()
        optimizer.step()
        with torch.no_grad():
            logging.info("##Snl: calculating learning rate")
            qq1 = F.softmax(snl_forward(net, images, False), dim=1) + self.epsilon
            delta_pq, delta_qq1 = pp-qq0, qq1-qq0

            logging.info("##Snl: calculating eta_analytic_n2")
            norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro')
            eta2_raw, cos_phi = snl_eta(self.eta1, delta_pq, delta_qq1, norm_pq, norm_qq1, self.epsilon, self.beta_min, self.do_logging)
            eta2 = eta2_raw * self.alpha_epoch*alpha_momentum
            if self.do_logging:
                logging.info("##Snl: alpha_epoch={}, alpha_momentum={}, eta2={}".format(self.alpha_epoch, alpha_momentum, eta2))

            logging.info("##Snl: shifting params to the rest of step")
            eta2_shift = eta2 - self.eta1
            for group in optimizer.param_groups:
                params: List[Tensor] = []
                grads: List[Tensor] = []
                momentum_buffer_list: List[Optional[Tensor]] = []

                has_sparse_grad = optimizer._init_group(
                    group, params, grads, momentum_buffer_list
                )
                grad_norm2_squared, buffer_norm2_squared = 0.0, 0.0
                    #torch.tensor(0.0).to(meta.device), torch.tensor(0.0).to(meta.device)
                for num, param in enumerate(params):
                    grad, momentum_buffer = grads[num], momentum_buffer_list[num]
                    vbuff = grad if group["momentum"] == 0 else momentum_buffer
                    param.add_(vbuff, alpha=-eta2_shift)
                    if self.do_calc_grad_norm2:
                        grad_norm2_squared += (grad**2).sum().item() #Check if .item() fine for performance
                        buffer_norm2_squared += (momentum_buffer**2).sum().item()

            logging.info("####Snl: step finish, returning step_result")
        return StepResult( eta2, norm_pq, norm_qq1, cos_phi, self.alpha_epoch*alpha_momentum,\
                          grad_norm2_squared, buffer_norm2_squared)
