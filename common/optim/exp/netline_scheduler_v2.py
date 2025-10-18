import torch
from torch import Tensor
from torch.linalg import norm
import torch.nn.functional as F
from typing import List, Optional
#from torch.optim.optimizer import _use_grad_for_differentiable

import math
import logging
from random import uniform

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
    
def eta(eta_test, delta_pq, delta_qq, norm_pq, norm_qq, epsilon, beta_min, do_logging):
    #with torch.no_grad():
    cos_phi = torch.sum(delta_pq*delta_qq)/(norm_pq*norm_qq + epsilon)
    eta_next = norm_pq*cos_phi*eta_test/torch.maximum(norm_qq, beta_min)
    if do_logging:
        logging.info("##Snl: cos(pp^qq)={}, norm_pq={}, norm_qq={}, eta_test={}, eta_raw={}, beta_min={}"\
                    .format(cos_phi, norm_pq, norm_qq, eta_test, eta_next, beta_min))
    return eta_next, cos_phi

def tn(value, device):
    if torch.is_tensor(value):
        return value.detach().clone().to(device)
    return torch.tensor(value).to(device)

class StepResult:
    def __init__(self, eta, pq_norm=0.0, qq_norm=0.0, cos_phi=0.0, alpha = None, grad_norm2_squared=None, accum_norm2_squared=None, \
                 lr_adjustment=False):
        self.eta = eta
        self.pq_norm = pq_norm
        self.qq_norm = qq_norm
        self.cos_phi = cos_phi
        self.alpha = alpha
        self.grad_norm2_squared = grad_norm2_squared
        self.accum_norm2_squared = accum_norm2_squared
        self.lr_adjustment = lr_adjustment

class NetLineStepLR:

    #Values for lr, momentum and weight_decay are set externally in optimiser
    def __init__(self, net, optimizer, meta, foreach=False):
        self.net = net
        self.optimizer = optimizer
        self.meta = meta
        self.foreach = foreach

        self.eta1 = optimizer.param_groups[0]['lr'] #1st step eta-size
        self.momentum = optimizer.param_groups[0]['momentum']
        self.alpha_epoch = 0.9 #eta multiplier
        self.beta_min = tn(0.00001, meta.device) #min for eta denom for the eta-calculation stability
        self.epsilon = 1e-9

        self.dropout_mode = False #Set true if the net uses dropout layers
        self.do_logging = False #Is additional params logging performed or not, the logging may affect performance
        self.shortening_lr_for_momentum = False #If momentum > 0, shorten lr by theoretical ratio |g|/|v|
        self.shortening_lr_adjustment_probe = 0.0 #If shortening_lr_for_momentum = True, probe to adjust for real |g|/|v|
        self.shortening_lr_adjustment_samplesize = 200

        self._alpha_momentum = tn(1.0, meta.device)
        self._alpha_recalc = False
        self._alpha_momentum_queue = None
        self._alpha_momentum_queue_pos = 0

    def init_alpha_momentum(self):
        meta = self.meta
        self._alpha_momentum, self._alpha_recalc = tn(1.0, meta.device), False
        if self.momentum > 0.0 and self.shortening_lr_for_momentum:
            self._alpha_momentum = tn(math.sqrt(1-self.momentum**2), meta.device)
            if self.shortening_lr_adjustment_probe > 0.0:
                self._alpha_momentum_queue = \
                    torch.full((self.shortening_lr_adjustment_samplesize,), self._alpha_momentum).to(meta.device)

    def _recalc_alpha_momentum(self):
        if self._alpha_recalc == False:
            return self._alpha_momentum
        
        meta = self.meta
        self._alpha_momentum = tn(1.0, meta.device)
        if self.momentum > 0.0 and self.shortening_lr_for_momentum:
            self._alpha_momentum = tn(math.sqrt(1-self.momentum**2), meta.device)
            if self.shortening_lr_adjustment_probe > 0.0:
                alpha_fact = torch.mean(self._alpha_momentum_queue)
                self._alpha_momentum = tn(min((2.0*self._alpha_momentum - alpha_fact), alpha_fact), meta.device)

        self._alpha_recalc = False
        return self._alpha_momentum

    def step(self, labels, images):
        net = self.net
        optimizer = self.optimizer

        if self.dropout_mode and net.training:
            raise ValueError("For dropout_mode == True net.training must be False")
        logging.info("##Snl: Step start calculating logits and qq0")
        net.zero_grad()
        logitsG = snl_forward(net, images, self.dropout_mode) ## new gradient with dropout is generated here (1*)
        logging.info("##Snl: calculating criterion")
        loss = F.cross_entropy(logitsG, labels, reduction='mean')
        logging.info("##Snl: performing small step")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits0 = (logitsG if self.dropout_mode == False else snl_forward(net, images, False)) #.detach().clone()
            return self._internal_step(labels, images, logits0)


    #@_use_grad_for_differentiable
    def _internal_step(self, labels, images, logits0):
        net = self.net
        meta = self.meta
        optimizer = self.optimizer

        logging.info("##Snl: , calculating pp")
        pp = F.one_hot(labels, meta.output_dim) #.to(meta.device)
        qq0 = F.softmax(logits0, dim=1) + self.epsilon ## all qqxx calculated with dropout off
        logging.info("##Snl: calculating learning rate")
        qq1 = F.softmax(snl_forward(net, images, False), dim=1) + self.epsilon #.detach().clone()
        delta_pq, delta_qq1 = pp-qq0, qq1-qq0

        logging.info("##Snl: calculating eta_analytic_n2")
        alpha_momentum = self._recalc_alpha_momentum()
        norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro') #math.sqrt((delta_pq**2).sum().item()), math.sqrt((delta_qq1**2).sum().item()) #
        eta2_raw, cos_phi = eta(self.eta1, delta_pq, delta_qq1, norm_pq, norm_qq1, self.epsilon, self.beta_min, self.do_logging)
        eta2 = eta2_raw * self.alpha_epoch*alpha_momentum
        if self.do_logging:
            logging.info("##Snl: alpha_epoch={}, alpha_momentum={}, eta2={}".format(self.alpha_epoch, alpha_momentum, eta2))
        
        logging.info("##Snl: shifting params to the rest of step")
        eta2_shift = eta2 - self.eta1
        grad_norm2_squared, buffer_norm2_squared = torch.tensor(0.0).to(meta.device), torch.tensor(0.0).to(meta.device)
        do_lr_adjustment = False
        if self.momentum > 0.0 and self.shortening_lr_for_momentum and self.shortening_lr_adjustment_probe > 0.0:
            do_lr_adjustment = uniform(0, 1) < self.shortening_lr_adjustment_probe

        for group in optimizer.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = optimizer._init_group(
                group, params, grads, momentum_buffer_list
            )

            if self.foreach == False:
                for num, param in enumerate(params):
                    grad, momentum_buffer = grads[num], momentum_buffer_list[num]
                    buffer_x_shift = None
                    if group["momentum"] == 0:
                        buffer_x_shift = grad.mul(-eta2_shift)
                    else:
                        buffer_x_shift = momentum_buffer.mul(-eta2_shift)
                    param.add_(buffer_x_shift)
                    #TODO: norm calculation drops performance, slow operation
                    if do_lr_adjustment:
                        grad_norm2_squared += (grad**2).sum() #.item() #Check if .item() fine for performance
                        buffer_norm2_squared += (momentum_buffer**2).sum() #.item()

            else:
                buffers_x_shift = None
                if group["momentum"] == 0:
                    buffers_x_shift = torch._foreach_mul(grads, -eta2_shift)
                #    torch._foreach_add_(params, grads, alpha=-eta2_shift)
                else:
                    buffers_x_shift = torch._foreach_mul(momentum_buffer_list, -eta2_shift)
                #    torch._foreach_add_(params, momentum_buffer_list, alpha=-eta2_shift)
                torch._foreach_add_(params, buffers_x_shift)

                #TODO: norm calculation drops performance, slow operation
                if do_lr_adjustment:
                    for grad_ in torch._foreach_pow(grads, 2.0):
                        grad_norm2_squared += grad_.sum() #.item()
                    for momentum_ in torch._foreach_pow(momentum_buffer_list, 2.0):
                        buffer_norm2_squared += momentum_.sum() #.item()

        if do_lr_adjustment:
            self._alpha_momentum_queue[self._alpha_momentum_queue_pos] = torch.sqrt(grad_norm2_squared/buffer_norm2_squared)
            self._alpha_momentum_queue_pos = self._alpha_momentum_queue_pos + 1\
                if self._alpha_momentum_queue_pos < self.shortening_lr_adjustment_samplesize - 1 else 0
            self._alpha_recalc = True

        logging.info("####Snl: step finish, returning step_result")
        return StepResult( eta2, norm_pq, norm_qq1, cos_phi, self.alpha_epoch*alpha_momentum,\
                            grad_norm2_squared, buffer_norm2_squared, do_lr_adjustment)
