import torch
#from torch import Tensor
from torch.linalg import norm
import torch.nn.functional as F
#from typing import List, Optional
from torch.optim.optimizer import _use_grad_for_differentiable

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
    def __init__(self, net, optimizer, meta):
        self.net = net
        self.optimizer = optimizer
        self.meta = meta

        #self.eta1 = optimizer.param_groups[0]['lr'] #1st step eta-size
        self.alpha_epoch = 0.9 #eta multiplier
        #self.beta_min = torch.tensor(0.00001).to(meta.device) #min for eta denom for the eta-calculation stability
        #self.epsilon = 1e-9

        self.dropout_mode = False #Set true if the net uses dropout layers
        self.do_logging = False #Is additional params logging performed or not, the logging may affect performance
        #self.do_calc_grad_norm2 = False #Is norm2 squared of gradient calculated or not, the calculation may affect performance
        #self.do_shorten_lr_for_momentum = False #If momentum > 0, shorten lr by theoretical ratio |g|/|v|

        self.defaults = dict(
            differentiable=False,
        )

    def step(self, labels, images):
        net = self.net
        meta = self.meta
        optimizer = self.optimizer

        if self.dropout_mode and net.training:
            raise ValueError("For dropout_mode == True net.training must be False")
        logging.info("##Snl: Step start calculating logits and qq0")
        net.zero_grad()
        logitsG = snl_forward(net, images, self.dropout_mode) ## new gradient with dropout is generated here (1*)
        logging.info("##Snl: calculating criterion and gradient")
        loss = F.cross_entropy(logitsG, labels, reduction='mean')
        loss.backward()
        with torch.no_grad():
            logging.info("##Snl: parameters are preparing")
            optimizer.pp = F.one_hot(labels, meta.output_dim).to(self.meta.device)
            optimizer.logits0 = logitsG if self.dropout_mode == False else snl_forward(net, images, False)
            optimizer.alpha_epoch = self.alpha_epoch
            def closure():
                return net.forward(images)

            logging.info("##Snl: step-start")
            eta2, norm_pq, norm_qq1, cos_phi, alpha_final, grad_norm2_squared, buffer_norm2_squared =\
            optimizer.step(closure)
            logging.info("####Snl: step-finish, returning step_result")
            return StepResult( eta2, norm_pq, norm_qq1, cos_phi, alpha_final, grad_norm2_squared, buffer_norm2_squared)        
