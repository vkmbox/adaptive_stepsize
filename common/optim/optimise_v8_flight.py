import torch
from torch import nn
from torch.linalg import norm
import torch.nn.functional as F

import math
import logging

def labels_to_softhot(true_labels, meta):
    with torch.no_grad():
        return F.one_hot(true_labels, meta.output_dim).to(meta.device)

class ParameterProcessor:
    def __init__(self, device, foreach=False):
        #self.theta_current = {}
        self.delta_current = {}
        self.grad_current = {}
        self.device = device
        self.foreach = foreach

    def is_delta_empty(self):
        return len(self.delta_current) <= 0
    
    #def is_theta_empty(self):
    #    return len(self.theta_current) <= 0

    #def save_theta(self, model):
    #    with torch.no_grad():
    #        for name, param in model.named_parameters():
    #            self.theta_current[name] = param.detach().clone()

    #TODO: slow procedure
    #theta_current must be saved at the current iteration's beginning, delta must be accumulated from previous iterations
    def set_theta(self, model, eta):
        with torch.no_grad():
            params, deltas = list(), list()
            for name, param in model.named_parameters():
                delta = self.delta_current.get(name, None)
                #theta_base = self.theta_current[name]
                if self.foreach == False:
                    param.data.add_(delta, alpha=-eta)
                else:
                    params.append(param.data)
                    deltas.append(delta)

            if self.foreach == True:
                if isinstance(eta, torch.Tensor) and torch._utils.is_compiling():
                    deltas_x_lr = torch._foreach_mul(deltas, -eta)
                    torch._foreach_add_(params, deltas_x_lr)
                else:
                    torch._foreach_add_(params, deltas, alpha=-eta)

    #theta_current must be saved at the current iteration's beginning
    def save_delta_current(self, model, momentum, weight_decay = 0.0, calc_norm2_squared=False):
        with torch.no_grad():
            norm2_squared = torch.tensor(0.0).to(self.device)
            params, deltas, grads = list(), list(), list()
            ignore_foreach = False
            for name, param in model.named_parameters():
                delta = self.delta_current.get(name, None)
                grad = self.grad_current.get(name, None)
                if grad is None:
                    raise ValueError("No grad for param {}".format(name))
                if delta is None or momentum <= 0.0:
                    ignore_foreach = True
                    self.delta_current[name] = grad + weight_decay*param
                else:
                    if self.foreach == False:
                        delta.mul_(momentum).add_(grad, alpha=1).add_(param, alpha=weight_decay)
                    else:
                        params.append(param.data)
                        deltas.append(delta)
                        grads.append(grad)

                if self.foreach == True and ignore_foreach == False:
                    torch._foreach_mul_(deltas, momentum)
                    torch._foreach_add_(deltas, grads, alpha=1)
                    if weight_decay > 0:
                        torch._foreach_add_(deltas, params, alpha=weight_decay)

                #    self.delta_current[name] = grad + weight_decay*theta_base
                #else:
                #    self.delta_current[name] = momentum*delta + grad + weight_decay*theta_base

                if calc_norm2_squared:
                    norm2_squared += ((self.delta_current[name])**2).sum() #Check if .item() fine for performance

            return norm2_squared

    #Grad optionally multiplied by lambda
    def calc_autograd(self, model, loss, calc_norm2_squared=False):
        param_buffer ={}
        for name, param in model.named_parameters():
            param_buffer[name] = param
        logging.info("##Autograd start")
        df = torch.autograd.grad(loss, param_buffer.values(), create_graph=False, retain_graph=False)
        logging.info("##Autograd finish")
        with torch.no_grad():
            norm2_squared = torch.tensor(0.0).to(self.device)
            ii = 0
            for name in param_buffer:
                grad = df[ii] #.detach().clone()
                self.grad_current[name] = grad
                if calc_norm2_squared:
                    norm2_squared += ((grad)**2).sum() #Check if .item() fine for performance
                ii += 1
            return norm2_squared

class StepResult:
    def __init__(self, eta, pq_norm=0.0, qq_norm=0.0, cos_phi=0.0, kappa_avg=1.0, kappa_min=1.0\
                 , alpha = None, grad_norm2_squared=None, accum_norm2_squared=None, cos_phi_sample=None, ratio_sample=None):
        self.eta = eta
        self.pq_norm = pq_norm
        self.qq_norm = qq_norm
        self.cos_phi = cos_phi
        self.kappa_avg = kappa_avg
        self.kappa_min = kappa_min
        self.grad_norm2_squared = grad_norm2_squared
        self.accum_norm2_squared = accum_norm2_squared
        self.cos_phi_sample = cos_phi_sample
        self.ratio_sample = ratio_sample
        self.alpha = alpha

class NetLineStepProcessorAbstract:
    def __init__(self, net, meta, device, foreach=False):
        self.net = net
        self.meta = meta
        self.device = device
        self.epsilon = 1e-9
        self.paramProcessor = ParameterProcessor(device, foreach)
        self.do_logging = False #Is additional params logging performed or not, the logging may affect performance
        self.do_calc_grad_norm2 = False #Is norm2 squared of gradient calculated or not, the calculation may affect performance
        self.shorten_grad_accumulated = False #If momentum > 0, shorten accum grad to the length of latest one via alpha

        self.alpha = 0.5 #eta multiplier
        self.beta_min = torch.tensor(0.00001).to(device) #min for eta denom for the eta-calculation stability
        self.kappa_step_pp = 0.0 #kappa step per trainpoint
        self.eta1 = 0.00001 #1st step eta-size
        self.tensor_zero = torch.tensor(0.0).to(device)

        self.dropout_mode = False #Set true if the net uses dropout layers

    def get_param(self, step_params, param_name, default):
        if step_params is None:
            return default
        return step_params.get(param_name, default)

    def eta_analytic_n2_onehot(self, eta_test, delta_pq, delta_qq, norm_pq, norm_qq):
        with torch.no_grad():
            cos_phi = torch.sum(delta_pq*delta_qq)/(norm_pq*norm_qq + self.epsilon)
            eta_next = norm_pq*cos_phi*eta_test/torch.maximum(norm_qq, self.beta_min)
            if self.do_logging:
                logging.info("##cos(pp^qq)={}, norm_pq={}, norm_qq={}, eta_test={}, eta_next_raw={}, beta_min={}"\
                            .format(cos_phi, norm_pq, norm_qq, eta_test, eta_next, self.beta_min))
            return eta_next, cos_phi

    def softmax(self, logits):
        with torch.no_grad():
            return (F.softmax(logits, dim=1) + self.epsilon) #.to(meta.device) torch.transpose(, 0, 1)

    def crossentropy_avg(self, pp, qq):
        with torch.no_grad():
            #return (F.nll_loss(torch.log(torch.transpose(qq, 0, 1)), true_labels, reduction='mean')).item()
            batch_size = pp.shape[0]
            return -torch.sum(torch.log(torch.sum(pp*qq, 1)))/batch_size

    #force_trainmode=True/False
    def do_forward(self, images, force_trainmode):
        net = self.net
        if force_trainmode == True:
            training = net.training
            net.train(True)
            logging.info("##--==Explicit train forward==--")
            logits = net.forward(images)
            net.train(training)
            return logits
        else:
            return net.forward(images)

class NetLineStepProcessor(NetLineStepProcessorAbstract):
    def __init__(self, net, meta, device, foreach=False):
        super().__init__(net, meta, device, foreach)
        #self.internal_criterion = nn.CrossEntropyLoss(reduction='none')

    def calc_criterion(self, logits, labels, pp):
        if self.kappa_step_pp <= 0.0:
            return F.cross_entropy(logits, labels, reduction='mean'), 1.0, 1.0
        else:
            with torch.no_grad():
                kappa_raw = 1.0 + self.kappa_step_pp*torch.sum(pp*F.log_softmax(logits, dim=1), dim=1)
                kappa = torch.maximum(kappa_raw, self.tensor_zero)
                kappa_avg, kappa_min = torch.mean(kappa), torch.min(kappa)
                if self.do_logging:
                    logging.info("##Kappa: avg={}, min={}, max={}".format(kappa_avg, kappa_min, torch.max(kappa)))

            return torch.mean(kappa*F.cross_entropy(logits, labels, reduction='none')), kappa_avg, kappa_min

    def step(self, labels, images, momentum = 0.0, weight_decay = 0.0):

        net = self.net
        meta = self.meta
        if self.dropout_mode and net.training:
            raise ValueError("For dropout_mode == True net.training must be False")
        logging.info("##Tmp: Calculating labels_to_softhot")
        pp = labels_to_softhot(labels, meta)
        logging.info("##Saving theta")
        #if self.paramProcessor.is_theta_empty():
        #    self.paramProcessor.save_theta(net)

        logging.info("##Calculating params-delta")
        net.zero_grad()
        logits = self.do_forward(images, self.dropout_mode) ## new gradient with dropout is generated here (1*)
        logging.info("##Calculating criterion")
        loss, kappa_avg, kappa_min = self.calc_criterion(logits, labels, pp)
        logging.info("##Criterion calculated")
        norm2_squared_latest = self.paramProcessor.calc_autograd(net, loss, self.do_calc_grad_norm2)
        logging.info("##Autograd calculated")
        norm2_squared_accumulated = self.paramProcessor.save_delta_current(net, momentum, weight_decay, self.do_calc_grad_norm2)
        logging.info("##Delta saved")
        #torch.sqrt(norm2_squared_latest/norm2_squared_accumulated).item() \
        alpha_shortening = math.sqrt(1-momentum**2) if momentum > 0.0 and self.shorten_grad_accumulated else 1.0
        with torch.no_grad():
            if self.dropout_mode:
                logits = self.do_forward(images, False) ## all qqxx calculated with dropout off
            #Eta-calculation
            qq0 = self.softmax(logits) #q(t)
            #Sample 
            logging.info("##Tmp: Before loss initial")
            #eta_curr = self.eta1 #small step-size
            self.paramProcessor.set_theta(net, self.eta1) #small step
            #logits1 = self.do_forward(images, False)
            qq1 = self.softmax(self.do_forward(images, False))
            delta_pq, delta_qq1 = pp-qq0, qq1-qq0

            logging.info("##Tmp: Before eta_analytic_n2")
            norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro')
            eta2_raw, cos_phi = self.eta_analytic_n2_onehot(self.eta1, delta_pq, delta_qq1, norm_pq, norm_qq1)
            eta2 = eta2_raw * self.alpha* alpha_shortening
            if self.do_logging:
                logging.info("##alpha={}, alpha_shortening={}, eta2={}".format(self.alpha, alpha_shortening, eta2))
            logging.info("##Tmp: After eta_analytic_n2")
            self.paramProcessor.set_theta(net, eta2 - self.eta1) #regular step
            logging.info("##Tmp:regular step finish")

            logging.info("##Tmp:finish")
            return StepResult( eta2, norm_pq, norm_qq1, cos_phi, kappa_avg, kappa_min, self.alpha*alpha_shortening\
                              , norm2_squared_latest, norm2_squared_accumulated)

