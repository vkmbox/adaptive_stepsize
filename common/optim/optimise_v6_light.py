import numpy as np

import torch
from torch import nn
from torch.linalg import norm
import torch.nn.functional as F

from common.optim.cubic import cubic_real_positive_smallest

import logging

def sign(number):
    return (-1.0 if number < 0.0 else 1.0)


def labels_to_softhot(true_labels, meta):
    with torch.no_grad():
        return torch.transpose(F.one_hot(true_labels, meta.output_dim), 0, 1).to(meta.device)

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

    #theta_current must be saved at the current iteration's beginning, delta must be accumulated from previous iterations
    def set_theta(self, model, momentum, eta, eta_scale = 1.0, weight_decay = 0.0):
        eta_w = eta * eta_scale
        with torch.no_grad():
            for name, param in model.named_parameters():
                grad = self.grad_current[name]
                delta = self.delta_current.get(name, None)
                if momentum <= 0.0 and eta_scale > 0.0:
                    param.data.copy_(self.theta_current[name] -eta_w * grad - eta_w*weight_decay*self.theta_current[name])
                elif eta_scale <= 0.0 and momentum > 0.0:
                    if delta is not None:
                        param.data.copy_(self.theta_current[name] + momentum * delta)
                    else:
                        param.data.copy_(self.theta_current[name])
                elif eta_scale > 0.0 and momentum > 0.0:
                    if delta is not None:
                        param.data.copy_(self.theta_current[name] + momentum * delta - eta_w * grad - eta_w*weight_decay*self.theta_current[name])
                    else:
                        param.data.copy_(self.theta_current[name] - eta_w * grad - eta_w*weight_decay*self.theta_current[name])
                else:
                    raise ValueError("eta_scale or momentum must be in interval(0., 1.). momentum={}, eta={}, eta_scale={}"\
                                     .format(momentum, eta, eta_scale))

    #theta_current must be saved at the current iteration's beginning
    def save_delta_current(self, momentum, eta, eta_scale = 1.0, weight_decay = 0.0):
        eta_w = eta * eta_scale
        with torch.no_grad():
            for name, grad in self.grad_current.items():
                delta = self.delta_current.get(name, None)
                if delta is None or momentum <= 0.0:
                    self.delta_current[name] = -eta_w * grad- eta_w*weight_decay*self.theta_current[name]
                else:
                    self.delta_current[name] = momentum * delta - eta_w * grad- eta_w*weight_decay*self.theta_current[name]

    #Grad optionally multiplied by lambda
    def calc_autograd(self, model, loss, lambda_dict=None, calc_norm2_squared=False):
        param_buffer ={}
        for name, param in model.named_parameters():
            param_buffer[name] = param
        logging.info("##Autograd start")
        df = torch.autograd.grad(loss, param_buffer.values())#, retain_graph=True, create_graph=True, allow_unused=True)
        logging.info("##Autograd finish")
        with torch.no_grad():
            norm2_squared = 0.0
            ii = 0
            for name in param_buffer:
                lambda_value = lambda_dict.get(name, 1.) if lambda_dict is not None else 1.
                grad = df[ii] #.detach().clone()
                if lambda_value != 1. :
                    grad = lambda_value * grad
                self.grad_current[name] = grad
                if calc_norm2_squared:
                    norm2_squared += ((grad)**2).sum()
                ii += 1
            return norm2_squared

class StepResult:
    def __init__(self, eta, eta2, quadtatic_ratio, acc_ratio, by_rate_ratio,\
                 ck_armiho=0.0, ck_wolf=0.0, pq_norm=0.0, qq_norm=0.0, qq2_norm=0.0, norm_base=0.0, cos_phi=0.0, cos_base=0.0, \
                 grad_norm2_squared=None, a0 = 0., a1 = 0., a1_2 = 0., a2 = 0., a3 = 0.):
        self.eta = eta
        self.eta2 = eta2
        self.quadtatic_ratio = quadtatic_ratio
        self.acc_ratio=acc_ratio
        self.by_rate_ratio = by_rate_ratio
        self.ck_armiho = ck_armiho
        self.ck_wolf = ck_wolf
        self.pq_norm = pq_norm
        self.qq_norm = qq_norm
        self.qq2_norm = qq2_norm
        self.norm_base = norm_base
        self.cos_phi = cos_phi
        self.cos_base = cos_base
        self.grad_norm2_squared = grad_norm2_squared
        self.a0 = a0
        self.a1 = a1
        self.a1_2 = a1_2
        self.a2 = a2
        self.a3 = a3

class NetLineStepProcessorAbstract:
    def __init__(self, net, meta, device, lbd_dict=None):
        self.net = net
        self.meta = meta
        self.device = device
        self.epsilon = 1e-9
        self.lbd_dict = lbd_dict
        self.paramProcessor = ParameterProcessor()

        self.training_mode = False
        self.do_logging = False
        self.calc_baseangle = False
        self.quadratic_step = False #is quadratic_step enabled
        self.calc_autoalpha = 0.0

        self.alpha = 0.5 #raw eta multiplier
        self.beta = 0.00025 #term in raw eta denom for the raw-eta stability
        self.gamma = 0.1 #cos-phi threshold
        self.delta = 0.25 #fraction of initial velocity to stop the learning rate
        self.eta1 = 0.0001 #1st step eta-size
        self.tensor_one = torch.tensor(1.0).to(device)
        #self.tensor_zero = torch.tensor(0.0).to(device)
        #self.tensor_true = torch.tensor(True).to(device)

    def get_param(self, step_params, param_name, default):
        if step_params is None:
            return default
        return step_params.get(param_name, default)

    def eta_analytic_n2_onehot(self, eta_test, delta_pq, delta_qq, norm_pq, norm_qq): #pp, qq0, qq_test):
        with torch.no_grad():
            #delta_pq, delta_qq = pp-qq0, qq_test-qq0
            #norm_pq, norm_qq = norm(delta_pq, ord='fro'), norm(delta_qq, ord='fro')
            cos_phi = torch.sum(delta_pq*delta_qq)/(norm_pq*norm_qq + self.epsilon)
            cos_factor = torch.min((cos_phi**2/self.gamma), self.tensor_one)
            alpha_val = 1/(10*cos_phi*self.calc_autoalpha) if self.calc_autoalpha > 0.0 else self.alpha
            eta_next = alpha_val*cos_factor*norm_pq*cos_phi*eta_test/(norm_qq + self.beta)
            #eta_next = self.alpha*torch.where(0.0 < eta_original and eta_original < self.delta and cos_factor == 1.0\
            #                                  , torch.sqrt(self.delta*eta_original), eta_original)
            if self.do_logging:
                logging.info("##cos(pp^qq)={}, norm_pq={}, norm_qq={}, eta_test={}, eta_next={}, alpha={}, beta={}, run_ratio={}"\
                            .format(cos_phi, norm_pq, norm_qq, eta_test, eta_next, self.alpha, self.beta, cos_factor))
            return eta_next, cos_phi

    def softmax(self, logits, meta):
        with torch.no_grad():
            return (F.softmax(torch.transpose(logits, 0, 1), dim=0) + self.epsilon) #.to(meta.device)
        
    def crossentropy_avg(self, pp, qq):
        with torch.no_grad():
            #return (F.nll_loss(torch.log(torch.transpose(qq, 0, 1)), true_labels, reduction='mean')).item()
            batch_size = pp.shape[1]
            return -torch.sum(torch.log(torch.sum(pp*qq, 0)))/batch_size
            #return loss.item()

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

    """
    step_params: ;
    """
    def step(self, labels, images, momentum = 0.0, nesterov = False, weight_decay = 0.0):
        net = self.net
        meta = self.meta
        #if net.training:
        #    raise ValueError("net.training must be False")
        logging.info("##Tmp: Calculating labels_to_softhot")
        pp = labels_to_softhot(labels, meta)
        logging.info("##Saving theta")
        self.paramProcessor.save_theta(net)

        qq_base = None
        if self.calc_baseangle:
            qq_base = self.softmax(self.do_forward(images, 'train'), meta)

        if momentum > 0.0 and nesterov == True and self.paramProcessor.is_delta_empty() == False:
            logging.info("##Setting grad theta")
            self.paramProcessor.set_theta(net, momentum, 0., 0., 0.)

        logging.info("##Calculating params-delta")
        net.zero_grad()
        logits = self.do_forward(images, 'toss') ## new gradient with dropout is generated here (1*)
        loss = self.criterion(logits, labels)
        grad_norm2_squared = self.paramProcessor.calc_autograd(net, loss, self.lbd_dict, True)
        logging.info("##Autograd calculated")
        with torch.no_grad():
            if self.training_mode:
                logits = self.do_forward(images, 'train') ## all qqxx calculated with dropout off
            #Eta-calculation
            qq0 = self.softmax(logits, meta) #q(t)
            logging.info("##Tmp: Before loss initial")
            #eta_curr = self.eta1 #small step-size
            self.paramProcessor.set_theta(net, momentum, self.eta1, 1.0, weight_decay) #small step
            logits1 = self.do_forward(images, 'train')
            qq1 = self.softmax(logits1, meta) #q(tets)
            logging.info("##Tmp: Before eta_analytic_n2")
            delta_pq, delta_qq1 = pp-qq0, qq1-qq0
            norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro')            
            eta2, cos_phi = self.eta_analytic_n2_onehot(self.eta1, delta_pq, delta_qq1, norm_pq, norm_qq1)
            logging.info("##Tmp: After eta_analytic_n2")
            eta_ratio = eta2/(self.eta1 + self.epsilon)
            loss_initial = self.crossentropy_avg(pp, qq0)
            diff_initial = -(torch.sum((pp/qq0)*(delta_qq1)))/pp.shape[1]
            diff_next = -(torch.sum((pp/qq1)*(delta_qq1)))/pp.shape[1]
            loss_next = self.crossentropy_avg(pp, qq1)
            ck1_armiho = (loss_next - self.epsilon - loss_initial)/(diff_initial+self.epsilon)
            ck1_wolf = ((diff_next) - self.epsilon)/(diff_initial+self.epsilon)
            if self.do_logging:
                logging.info("##--==On step 1: eta_curr={}, eta_next={}, ratio={}, loss_initial={}, loss_next={}, diff_initial={}, diff_next={}, ck1_armiho={}, ck1_wolf={} ==--"\
                            .format(self.eta1, eta2, eta_ratio, loss_initial, loss_next, diff_initial, diff_next, ck1_armiho, ck1_wolf))
            eta_curr = eta2
            self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0, weight_decay) #1st step
            logging.info("##Tmp:1st step finish")

            cos_base, norm_base, norm_qq2 = 0., 0., 0.
            quadtatic_ratio, acc_ratio, by_rate_ratio = 1., 1., 1.
            a0, a1, a2, a3, a1_1, a1_2, eta3_flex = 0., 0., 0., 0., 0., 0., 0.
            if self.quadratic_step and eta2 > 0.0:
                logits2 = self.do_forward(images, 'train')
                qq2 = self.softmax(logits2, meta) #q(t+1)
                delta_qq2 = qq2-qq0
                if self.calc_baseangle:
                    delta_base = qq0 - qq_base
                    norm_base = norm(delta_base, ord='fro')
                    norm_qq2 = norm(delta_qq2, ord='fro')
                    cos_base = torch.sum(delta_base*delta_qq2)/(norm_base*norm_qq2 + self.epsilon)

                diff_initial = -(torch.sum((pp/qq0)*(delta_qq2)))/pp.shape[1]
                diff_next = -(torch.sum((pp/qq2)*(delta_qq2)))/pp.shape[1]
                loss_next = self.crossentropy_avg(pp, qq2)
                ck1_armiho = (loss_next - self.epsilon - loss_initial)/(diff_initial+self.epsilon)
                ck1_wolf = ((diff_next) - self.epsilon)/(diff_initial+self.epsilon)
                if self.do_logging:
                    logging.info("##--==On step 2: loss_initial={}, loss_next={}, diff_initial={}, diff_next={}, ck1_armiho={}, ck1_wolf={} ==--"\
                                .format(loss_initial, loss_next, diff_initial, diff_next, ck1_armiho, ck1_wolf))

                #delta_eta1, delta_eta2 = delta_qq1/(self.eta1**2), delta_qq2/(eta2**2) solve_cubic
                aa = ((eta2/self.eta1)*delta_qq1 - (self.eta1/eta2)*delta_qq2)/(((eta2/self.eta1)-1.0)*delta_qq1 + self.epsilon)
                #aa = torch.where((0.0 < aa_raw) & (aa_raw < 2.0), aa_raw, 1.0)
                #logging.info("##Tmp:aa={}".format(aa))
                vv = delta_qq1*(1-aa)
                #Eta corrected for norm2
                a0 = -self.eta1 * torch.sum(delta_pq*delta_qq1*aa)
                #a0_alpha = torch.sum(delta_pq*delta_qq1*aa)/torch.sum(delta_pq*delta_qq1)
                a1_1, a1_2 = torch.sum((delta_qq1*aa)**2), - 2*torch.sum(delta_pq*vv)
                a1 = a1_1 + a1_2
                a2 = 3*torch.sum(delta_qq1*vv*aa)/self.eta1
                a3 = 2*torch.sum(vv**2)/(self.eta1**2)

                dterm = a2**2 - 3*a1*a3 
                acc_ratio = 1 - a2**2/dterm
                eta3_flex = -a2/(3*a3) if dterm < 0 else (dterm**(0.5)-a2)/(3*a3)
                #derivative must change sign from - to +; delta_pq includes minus with respect to formulas.
                eta3_quadtatic = torch.where(a0 >= 0.0, eta2, cubic_real_positive_smallest(a3, a2, a1, a0, eta2, self.do_logging))
                quadtatic_ratio = eta3_quadtatic/eta2

                #eta_corrected
                '''
                if a0 < 0.:
                    rate_threshold, eta3_by_rate = a0*self.delta, 0.
                    if (self.rate_at(a0, a1, a2, a3, eta2) < rate_threshold): #if rate at eta2 big enough
                        eta3_by_rate = eta2
                    else:
                        eta_left, eta_right = 0., eta2
                        if (dterm > 0. and eta3_flex > 0.): #no-monotonic rate case
                            if(rate_threshold < self.rate_at(a0, a1, a2, a3, eta3_flex)):
                                eta3_by_rate = eta3_flex
                            else:
                                eta_left = eta3_flex
                        if eta3_by_rate <= 0.:
                            for cnt in range(5):
                                eta_middle = (eta_left+eta_right)/2
                                if(rate_threshold < self.rate_at(a0, a1, a2, a3, eta_middle)):
                                    eta_right = eta_middle
                                else:
                                    eta_left = eta_middle
                            eta3_by_rate = (eta_left+eta_right)/2

                    if 0 < eta3_by_rate and eta3_by_rate < eta2:
                        eta_curr = eta3_by_rate
                        by_rate_ratio = eta3_by_rate/eta2
                '''
                #if 0< eta3_flex and eta3_flex < eta2 and (acc_ratio < 1 or self.delta < acc_ratio):
                #    eta_curr = eta3_flex

                #Eta corrected for norm1-active
                #delta_b = -delta_pq - 1
                '''
                b0 = -self.eta1 * torch.sum(pp*qq0*delta_qq1*aa)
                b1 = -torch.sum((pp*delta_qq1*aa)**2 + 2*pp*qq0*vv)
                b2 = -3*torch.sum(pp*delta_qq1*vv*aa)/self.eta1
                b3 = -2*torch.sum((pp*vv)**2)/(self.eta1**2)
                eta2_corrected_norm1 = cubic_real_positive_smallest(b3, b2, b1, b0, eta2)
                correction_ratio_norm1 = eta2_corrected_norm1/eta2
                '''

                if self.do_logging:
                    qq0a = torch.sum(qq0*pp, 0)
                    logging.info("##--==qq0-active: avg(qq0)={}, var(qq0)={}, min(qq0)={}, max(qq0)={}"\
                                 .format(torch.mean(qq0a), torch.var(qq0a), torch.min(qq0a), torch.max(qq0a)))                    
                    #logging.info("##--==On step 2: eta2_corrected_norm1={}, correction_ratio_norm1={}"\
                    #             .format(eta2_corrected_norm1, correction_ratio_norm1))

                    logging.info("##--==On step 2: avg(aa)={}, var(aa)={}, min(aa)={}, max(aa)={} ==--"\
                                 .format(torch.mean(aa), torch.var(aa), torch.min(aa), torch.max(aa)))

                    logging.info("##--==On step 2: eta3_quadtatic={}, quadtatic_ratio={}, acc_ratio={} ==--"\
                                 .format(eta3_quadtatic, quadtatic_ratio, acc_ratio))
            #    if 0.0 < correction_ratio and correction_ratio < self.quadratic_threshold:
            #        logits_curr, eta_curr = logits2, eta2_corrected
            #        logging.info("##Tmp:setting eta2_corrected={}".format(eta_curr))
            #        self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0, weight_decay) #2nd step

            '''
            if (self.get_param(step_params, 'check_eta2', False) == True or self.get_param(step_params, 'set_eta2', False) == True):
                logits12 = self.do_forward(images, 'train')
                qq12 = self.softmax(logits12, meta) #q(t+1)
                eta_next, _, _, _, _ = self.eta_analytic_n2_onehot(eta_curr, pp, qq0, qq12) #self.eta_analytic_n2(eta_curr, labels, qq0, qq12)
                eta_ratio = eta_next/(eta_curr + self.epsilon)
                logging.info("##--==On step 2 for eta_curr={} eta_next={} with ratio={} ==--".format(eta_curr, eta_next, eta_ratio))
                if (self.get_param(step_params, 'set_eta2', False) == True):
                    logits1, qq1, eta_curr = logits12, qq12, eta_next
                    self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0, weight_decay) #2st step
            '''
            #eta = eta_curr

            #eta_scale, logits = 1.0, logits1
            #logging.info("##Eta-value after conditions are applied: {}".format(eta*eta_scale))
            self.paramProcessor.save_delta_current(momentum, eta_curr, 1.0, weight_decay)
            logging.info("##Tmp:finish")
            #correction_ratio_effective = correction_ratio \
            #    if 0.0 < correction_ratio and correction_ratio < self.quadratic_threshold else 1.0
            return StepResult( eta_curr, eta2, quadtatic_ratio, acc_ratio, by_rate_ratio\
                              , ck1_armiho, ck1_wolf, norm_pq, norm_qq1, norm_qq2, norm_base, cos_phi, cos_base, grad_norm2_squared\
                              , a0, a1*eta2, a1_2*eta2, a2*(eta2**2), a3*(eta2**3))
        
    def rate_at(self, a0, a1, a2, a3, eta):
        return a0 + a1*eta + a2*(eta**2) + a3*(eta**3)

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
            self.paramProcessor.set_theta(net, momentum, self.eta1, 1.0) #small step
            zz1 = self.do_forward(images, 'train') #z(t+1)_test
            eta_next, eta_raw, norm_yz, norm_zz = self.eta_analytic_n2_onehot(self.eta1, yy, zz0, zz1)
            eta_ratio = eta_next/(self.eta1 + self.epsilon)
            logging.info("##--==On step 1 for eta_curr={} eta_next={} with ratio={} ==--".format(self.eta1, eta_next, eta_ratio))
            eta_curr = eta_next
            self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0) #1st step
            if (self.get_param(step_params, 'check_eta2', False) == True or self.get_param(step_params, 'set_eta2', False) == True):
                zz12 = self.do_forward(images, 'train') #z(t+1)
                eta_next, _, _, _ = self.eta_analytic_n2_onehot(eta_curr, yy, zz0, zz12)
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
        return StepResult(logits, eta*eta_scale, eta_raw, eta_ratio, ck1_armiho, ck1_wolf, norm_yz, norm_zz)
