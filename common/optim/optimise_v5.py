import math
import collections

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import logging

def sign(number):
    return (-1.0 if number < 0.0 else 1.0)

def reduce_to_active(matrix, pp):
    with torch.no_grad():
        return torch.sum(matrix*pp, 0)
    
def norm_fro(tensor, ord=2):
    with torch.no_grad():
        return math.pow((torch.sum(torch.abs(tensor)**ord)).item(), 1/ord)

def crossentropy_avg_old(pp, qq):
    with torch.no_grad():
        batch_size = pp.shape[1]
        loss = -torch.sum(pp * torch.log(qq))/batch_size
        return loss.item()

#TODO: function requires optimisation/rewriting
def labels_to_softhot(true_labels, meta):
    batch_size = true_labels.shape[0]
    #index = true_labels.unsqueeze(0)
    with torch.no_grad():
        data = torch.zeros(meta.output_dim, batch_size)
        data[true_labels, torch.arange(batch_size)] = 1.0
        #pp.scatter_(0, index, 1.)
        #for batch_num in range(batch_size):
        #    pp[true_labels[batch_num], batch_num] = 1.0
    return data.to(meta.device)

class ParameterProcessor:
    def __init__(self):
        self.theta_current = {}
        self.delta_current = {}
        self.grad_current = {}

    def is_delta_empty(self):
        return len(self.delta_current) <= 0

    def save_theta(self, model):
        with torch.no_grad():        
            for name, param in model.named_parameters():
                self.theta_current[name] = param.detach().clone()

    def set_theta(self, model, momentum, eta, eta_scale = 1.0):
        with torch.no_grad():
            for name, param in model.named_parameters():
                grad = self.grad_current[name]
                delta = self.delta_current.get(name, None)
                if momentum <= 0.0 and eta_scale > 0.0:
                    param.data.copy_(self.theta_current[name] -eta * eta_scale * grad)
                elif eta_scale <= 0.0 and momentum > 0.0:
                    if delta is not None:
                        param.data.copy_(self.theta_current[name] + momentum * delta)
                    else:
                        param.data.copy_(self.theta_current[name])
                elif eta_scale > 0.0 and momentum > 0.0:
                    if delta is not None:
                        param.data.copy_(self.theta_current[name] + momentum * delta - eta * eta_scale * grad)
                    else:
                        param.data.copy_(self.theta_current[name] - eta * eta_scale * grad)
                else:
                    raise ValueError("eta_scale or momentum must be in interval(0., 1.). momentum={}, eta={}, eta_scale={}"\
                                     .format(momentum, eta, eta_scale))

    def save_delta_current(self, momentum, eta, eta_scale = 1.0):
        with torch.no_grad():
            for name, grad in self.grad_current.items():
                delta = self.delta_current.get(name, None)
                if delta is None or momentum <= 0.0:
                    self.delta_current[name] = -eta * eta_scale * grad
                else:
                    self.delta_current[name] = momentum * delta - eta * eta_scale * grad

    #Grad optionally multiplied by lambda
    def calc_autograd(self, model, loss, lambda_dict=None):
        param_buffer ={}
        for name, param in model.named_parameters():
            param_buffer[name] = param
        logging.info("##Autograd start")
        df = torch.autograd.grad(loss, param_buffer.values())#, retain_graph=True, create_graph=True, allow_unused=True)
        logging.info("##Autograd finish")
        with torch.no_grad():
            #norm2_squared = 0.
            ii = 0
            for name in param_buffer:
                lambda_value = lambda_dict.get(name, 1.) if lambda_dict is not None else 1.
                grad = df[ii] #.detach().clone()
                if lambda_value != 1. :
                    grad = lambda_value * grad
                self.grad_current[name] = grad
                #norm2_squared += ((grad)**2).sum().item()
                ii += 1
            #return norm2_squared

class StepResult:
    def __init__(self, logits, eta, eta_raw, eta_ratio2, ck_armiho=0.0, ck_wolf=0.0, pq_norm=0.0, qq_norm=0.0):
        self.logits = logits
        self.eta = eta
        self.eta_raw = eta_raw
        self.eta_ratio2 = eta_ratio2
        self.ck_armiho = ck_armiho
        self.ck_wolf = ck_wolf
        self.pq_norm = pq_norm
        self.qq_norm = qq_norm

class NetLineStepProcessorAbstract:
    def __init__(self, net, meta, device, lbd_dict=None):
        self.net = net
        self.meta = meta
        self.device = device
        self.epsilon = 1e-9
        self.epsilon_criteria = 1e-9
        self.lbd_dict = lbd_dict
        self.paramProcessor = ParameterProcessor()
        self.training_mode = False

        self.eta_min = 0.000001
        self.eta_max = 1.
        self.alpha = 0.01
        self.beta = 0.001
        self.eta0 = 0.0001

    def get_param(self, step_params, param_name, default):
        if step_params is None:
            return default
        return step_params.get(param_name, default)

    def cos_phi(self, vector1, vector2, norm1, norm2):
        with torch.no_grad():
            cos_phi = (torch.sum(vector1*vector2)/(norm1*norm2 + self.epsilon)).item()
            return cos_phi

    def eta_analytic_n2(self, eta_test, pp, qq0, qq_test):
        with torch.no_grad():
            delta_pq, delta_qq = pp-qq0, qq_test-qq0
            norm_pq, norm_qq = norm_fro(delta_pq, ord=2), norm_fro(delta_qq, ord=2)
            cos_phi1 = self.cos_phi(delta_pq, delta_qq, norm_pq, norm_qq)
            logging.info("##cos(pp^qq)={}, norm_pq={}, norm_qq={}".format(cos_phi1, norm_pq, norm_qq))
            eta_next = sign(cos_phi1)*math.sqrt(abs(((norm_pq*cos_phi1*eta_test*self.alpha)/(norm_qq + self.beta))))
            logging.info("##Eta-value estimations: analytic={}".format(eta_next))
            return eta_next, norm_pq, norm_qq

    def softmax(self, logits, meta):
        with torch.no_grad():
            return (F.softmax(torch.transpose(logits, 0, 1), dim=0) + self.epsilon).to(meta.device)

    #TODO: function requires optimisation/rewriting
    def crossentropy_avg(self, pp, qq):
        with torch.no_grad():
            #return (F.nll_loss(torch.log(torch.transpose(qq, 0, 1)), true_labels, reduction='mean')).item()
            batch_size = pp.shape[1]
            loss = -torch.sum(torch.log(torch.sum(pp*qq, 0)))/batch_size
            return loss.item()

    def eta_bounded(self, eta):
        if abs(eta) > self.eta_max:
            logging.info("##Eta-value is reduced from {} to {}".format(eta, self.eta_max*sign(eta)))
            eta = self.eta_max*sign(eta)
        if abs(eta) < self.eta_min:
            logging.info("##Eta-value is increased from {} to {}".format(eta, self.eta_min*sign(eta)))
            eta = self.eta_min*sign(eta)
        return eta

    #dropout_mode = 'eval' #toss/train/eval
    def do_forward(self, images, dropout_mode):
        net = self.net
        if self.training_mode and dropout_mode == 'toss':
            training = net.training
            net.train(True)
            logging.info("##--==Explicit train forward==--")
            logits = net.forward(images)
            net.train(training)
            return logits
        else:
            return net.forward(images)

class NetLineStepProcessor(NetLineStepProcessorAbstract):
    def __init__(self, net, criterion, meta, device, lbd_dict=None):
        super().__init__(net, meta, device, lbd_dict) 
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    #Armiho: Loss(θ+α) <= c1*α*∇Loss(θ) + Loss(θ)
    #Wolf: |∇Loss(θ+α)| <= c2*|∇Loss(θ)|
    #0<c1<c2<1
    #Additional condition: ∇Loss(θ+α) <= c3*|∇Loss(θ)| , 0<c3<1 (Significant loss growth at the final point is unacceptable)
        self.c1 = 0.000001
        self.c2 = 0.999999
        self.c3 = 0.25
        self.armiho_beta = 0.5

    """
    step_params: check_armiho=True/False (default False); check_additional=True/False (default False);
                 check_eta2=True/False (default False); set_eta2=True/False (default False)
    """
    def step(self, labels, images, momentum = 0.0, nesterov = False, step_params = None):
        net = self.net
        meta = self.meta
        #if net.training:
        #    raise ValueError("net.training must be False")
        logging.info("##Tmp: Calculating labels_to_softhot")
        pp = labels_to_softhot(labels, meta)
        logging.info("##Saving theta")
        self.paramProcessor.save_theta(net)

        if momentum > 0.0 and nesterov == True and self.paramProcessor.is_delta_empty() == False:
            logging.info("##Setting grad theta")
            self.paramProcessor.set_theta(net, momentum, 0., 0.)

        logging.info("##Calculating params-delta")
        net.zero_grad()
        logits = self.do_forward(images, 'toss') ## new dropout is generated here (1*)
        loss = self.criterion(logits, labels)
        self.paramProcessor.calc_autograd(net, loss, self.lbd_dict)
        logging.info("##Autograd calculated")
        with torch.no_grad():
            if self.training_mode:
                logits = self.do_forward(images, 'train') ## all qqxx calculated with dropout off
            #Eta-calculation
            qq0 = self.softmax(logits, meta) #q(t)
            logging.info("##Tmp: Before loss initial")
            loss_initial = self.crossentropy_avg(pp, qq0)
            logging.info("##Loss initial:{}".format(loss_initial)) #, crossentropy_avg_old(pp, qq0)))
            eta_curr = self.eta0 #small step-size
            self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0) #small step
            logits1 = self.do_forward(images, 'train')
            qq1 = self.softmax(logits1, meta) #q(t+1)
            logging.info("##Tmp: Before eta_analytic_n2")
            eta_next, norm_pq, norm_qq = self.eta_analytic_n2(eta_curr, pp, qq0, qq1)
            logging.info("##Tmp: After eta_analytic_n2")
            eta_ratio = eta_next/(eta_curr + self.epsilon)
            logging.info("##--==On step 1 for eta_curr={} eta_next={} with ratio={} ==--".format(eta_curr, eta_next, eta_ratio))
            eta_curr = eta_next
            self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0) #1st step
            if (self.get_param(step_params, 'check_eta2', False) == True or self.get_param(step_params, 'set_eta2', False) == True):
                logits12 = self.do_forward(images, 'train')
                qq12 = self.softmax(logits12, meta) #q(t+1)
                eta_next, _, _ = self.eta_analytic_n2(eta_curr, pp, qq0, qq12)
                eta_ratio = eta_next/(eta_curr + self.epsilon)
                logging.info("##--==On step 2 for eta_curr={} eta_next={} with ratio={} ==--".format(eta_curr, eta_next, eta_ratio))
                if (self.get_param(step_params, 'set_eta2', False) == True):
                    logits1, qq1, eta_curr = logits12, qq12, eta_next
                    self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0) #2st step                    

            eta = self.eta_bounded(eta_curr)
            if eta != eta_curr:
                self.paramProcessor.set_theta(net, momentum, eta, 1.0)
                logits1 = self.do_forward(images, 'train')
                qq1 = self.softmax(logits1, meta)

        eta_scale, logits, ck1_armiho, ck1_wolf = 1.0, logits1, 1.0, 0.0
        if (self.get_param(step_params, 'check_armiho', True) == True or self.get_param(step_params, 'check_additional', True) == True):
            eta_scale, logits, ck1_armiho, ck1_wolf = self.step_reduction(images, pp, qq0, qq1, logits1, loss_initial, momentum, eta, step_params)

        logging.info("##Eta-value after conditions are applied: {}".format(eta*eta_scale))
        self.paramProcessor.save_delta_current(momentum, eta, eta_scale)
        return StepResult(logits, eta*eta_scale, eta_curr, eta_ratio, ck1_armiho, ck1_wolf, norm_pq, norm_qq)

    def step_reduction(self, images, pp, qq0, qq1, logits1, loss_initial, momentum, eta, step_params):
        net = self.net
        meta = self.meta
        logits_k = None
        eta_scale = 1.0
        with torch.no_grad():
            diff_initial = (torch.sum((pp/qq0)*(qq0-qq1))).item()/pp.shape[1]
            while eta_scale > 0.001:
                if eta_scale == 1.0:
                    logits_k = logits1
                else:
                    self.paramProcessor.set_theta(net, momentum, eta, eta_scale)
                    logits_k = self.do_forward(images, 'train') ## no new dropout generated here, a generated in (1*) must be used
                qq = self.softmax(logits_k, meta)
                loss_k = self.crossentropy_avg(pp, qq)
                diff_k = (torch.sum((pp/qq)*(qq0-qq1))).item()/pp.shape[1]
                #condition_armiho = loss_k - self.epsilon_criteria <= initialLoss + self.c1*eta_scale*diff_initial
                #condition_wolf = abs(diff_k) - self.epsilon_criteria <= self.c2*abs(diff_initial)
                ck1_armiho = (loss_k - self.epsilon_criteria - loss_initial)/(eta_scale*diff_initial+self.epsilon) #>=self.c1 when diff_initial < 0
                ck1_wolf = (abs(diff_k) - self.epsilon_criteria)/abs(diff_initial+self.epsilon) #<= self.c2
                condition_armiho = diff_initial < 0 and ck1_armiho >= self.c1
                condition_wolf = ck1_wolf <= self.c2
                condition_additional = diff_k - self.epsilon_criteria <= self.c3*abs(diff_initial)
                logging.info("##Step with eta_scale: {}, loss = {}, df(0) = {}, df({}) = {}"\
                             .format(eta_scale, loss_k, diff_initial, eta_scale, diff_k))
                logging.info("##Conditions: armiho = {}, wolf = {}, additional = {}, armiho_k={}, wolf_k={}"\
                             .format(condition_armiho, condition_wolf, condition_additional, ck1_armiho, ck1_wolf))
                if (self.get_param(step_params, 'check_armiho', True) == False or condition_armiho) \
                    and (self.get_param(step_params, 'check_additional', True) == False or condition_additional):
                    break
                eta_scale = eta_scale*self.armiho_beta
            
            return eta_scale, logits_k, ck1_armiho, ck1_wolf

class NetLineStepProcessorMSE(NetLineStepProcessorAbstract):
    def __init__(self, net, criterion, meta, device, lbd_dict=None):
        super().__init__(net, meta, device, lbd_dict) 
        self.criterion = criterion if criterion is not None else nn.MSELoss()

    """
    step_params: check_eta2=True/False (default False); set_eta2=True/False (default False)
    """
    def step(self, labels, images, momentum = 0.0, nesterov = False, step_params = None):
        net = self.net
        meta = self.meta
        xx, yy = images, labels
        self.paramProcessor.save_theta(net)

        if momentum > 0.0 and nesterov == True and self.paramProcessor.is_delta_empty() == False:
            self.paramProcessor.set_theta(net, momentum, 0., 0.)

        logging.info("##Calculating params-delta")
        net.zero_grad()
        zz = self.do_forward(xx, 'toss') ## new dropout is generated here (1*)
        loss = self.criterion(zz, yy)
        self.paramProcessor.calc_autograd(net, loss, self.lbd_dict)
        logging.info("##Autograd calculated")
        with torch.no_grad():
            if self.training_mode:
                zz = self.do_forward(xx, 'train') ## all qqxx calculated with dropout off
            #Eta-calculation
            zz0 = zz
            loss_initial = self.criterion(yy, zz0)
            logging.info("##Loss initial:{}".format(loss_initial))
            eta_curr = self.eta0 #small step-size
            self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0) #small step
            zz1 = self.do_forward(images, 'train') #z(t+1)_test
            eta_next, norm_yz, norm_zz = self.eta_analytic_n2(eta_curr, yy, zz0, zz1)
            eta_ratio = eta_next/(eta_curr + self.epsilon)
            logging.info("##--==On step 1 for eta_curr={} eta_next={} with ratio={} ==--".format(eta_curr, eta_next, eta_ratio))
            eta_curr = eta_next
            self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0) #1st step
            if (self.get_param(step_params, 'check_eta2', False) == True or self.get_param(step_params, 'set_eta2', False) == True):
                zz12 = self.do_forward(images, 'train') #z(t+1)
                eta_next, _, _ = self.eta_analytic_n2(eta_curr, yy, zz0, zz12)
                eta_ratio = eta_next/(eta_curr + self.epsilon)
                logging.info("##--==On step 2 for eta_curr={} eta_next={} with ratio={} ==--".format(eta_curr, eta_next, eta_ratio))
                if (self.get_param(step_params, 'set_eta2', False) == True):
                    zz1, eta_curr = zz12, eta_next
                    self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0) #2st step                    

            eta = self.eta_bounded(eta_curr)
            if eta != eta_curr:
                self.paramProcessor.set_theta(net, momentum, eta, 1.0)
                zz1 = self.do_forward(images, 'train')

        eta_scale, logits, ck1_armiho, ck1_wolf = 1.0, zz1, 1.0, 0.0
        #Armiho-check is not implemented for MSE-loss

        logging.info("##Eta-value after conditions are applied: {}".format(eta*eta_scale))
        self.paramProcessor.save_delta_current(momentum, eta, eta_scale)
        return StepResult(logits, eta*eta_scale, eta_curr, eta_ratio, ck1_armiho, ck1_wolf, norm_yz, norm_zz)
