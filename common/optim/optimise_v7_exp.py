import torch
from torch import nn
from torch.linalg import norm
import torch.nn.functional as F

import math
import logging

def labels_to_softhot(true_labels, meta):
    with torch.no_grad():
        return F.one_hot(true_labels, meta.output_dim).to(meta.device) #torch.transpose( , 0, 1)

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

    #TODO: slow procedure
    #theta_current must be saved at the current iteration's beginning, delta must be accumulated from previous iterations
    def set_theta(self, model, momentum, eta, eta_scale = 1.0, weight_decay = 0.0):
        logging.info("##set_theta:momentum={}, eta={}, eta_scale={}, weight_decay={}".format(momentum, eta, eta_scale, weight_decay))
        eta_w = eta * eta_scale
        with torch.no_grad():
            for name, param in model.named_parameters():
                grad = self.grad_current[name]
                delta = self.delta_current.get(name, None)
                theta_base = self.theta_current[name]
                #if eta_scale > 0.0:
                #    param.data.copy_(theta_base + eta_w * delta)
                if eta_scale >= 0.0 and momentum <= 0.0:
                    param.data.copy_((1-eta_w*weight_decay)*theta_base - eta_w*grad)
                #elif eta_scale <= 0.0 and momentum > 0.0:
                #    if delta is not None:
                #        param.data.copy_(theta_base + momentum * delta)
                #    else:
                #        param.data.copy_(theta_base)
                elif eta_scale >= 0.0 and momentum > 0.0:
                    #param_data = (1-eta_w*weight_decay)*theta_base - eta_w*grad
                    if delta is not None:
                        #param_data += eta_w*momentum*delta
                        param.data.copy_((1-eta_w*weight_decay)*theta_base - eta_w*grad - momentum*delta)
                    else:
                        param.data.copy_((1-eta_w*weight_decay)*theta_base - eta_w*grad)
                    #param.data.copy_(param_data)
                else:
                    raise ValueError("eta_scale must be in interval[0., 1.). momentum={}, eta={}, eta_scale={}"\
                                     .format(momentum, eta, eta_scale))

    #theta_current must be saved at the current iteration's beginning
    def save_delta_current(self, momentum, eta, eta_scale = 1.0, weight_decay = 0.0, calc_norm2_squared=False):
        logging.info("##save_delta_current:momentum={}, eta={}, eta_scale={}, weight_decay={}".format(momentum, eta, eta_scale, weight_decay))
        eta_w = eta * eta_scale
        with torch.no_grad():
            norm2_squared = 0.0
            for name, grad in self.grad_current.items():
                delta = self.delta_current.get(name, None)
                theta_base = self.theta_current[name]
                if delta is None or momentum <= 0.0:
                    self.delta_current[name] = eta_w * grad+ eta_w*weight_decay*theta_base
                    #self.delta_current[name] = -grad- weight_decay*theta_base
                else:
                    self.delta_current[name] = eta_w * grad+ eta_w*weight_decay*theta_base + momentum * delta
                    #self.delta_current[name] = momentum * delta - grad- weight_decay*theta_base

                if calc_norm2_squared:
                    norm2_squared += ((self.delta_current[name])**2).sum().item() #Check if .item() fine for performance

            return norm2_squared

    #Grad optionally multiplied by lambda
    def calc_autograd(self, model, loss, calc_norm2_squared=False):
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
                grad = df[ii] #.detach().clone()
                self.grad_current[name] = grad
                if calc_norm2_squared:
                    norm2_squared += ((grad)**2).sum().item() #Check if .item() fine for performance
                ii += 1
            return norm2_squared

class StepResult:
    def __init__(self, eta, ck_armiho=0.0, pq_norm=0.0, qq_norm=0.0, cos_phi=0.0, kappa_avg=1.0, kappa_min=1.0\
                 , alpha = None, grad_norm2_squared=None, accum_norm2_squared=None, cos_phi_sample=None, ratio_sample=None):
        self.eta = eta
        self.ck_armiho = ck_armiho
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
    def __init__(self, net, meta, device):
        self.net = net
        self.meta = meta
        self.device = device
        self.epsilon = 1e-9
        self.paramProcessor = ParameterProcessor()
        self.do_logging = False #Is additional params logging performed or not, the logging may affect performance
        self.do_calc_armiho = False #Is armiho coeff calculated or not, the calculation may affect performance
        self.do_calc_grad_norm2 = False #Is norm2 squared of gradient calculated or not, the calculation may affect performance

        self.alpha = 0.5 #eta multiplier
        self.beta_min = torch.tensor(0.00025).to(device) #min for eta denom for the eta-calculation stability
        self.kappa_step_pp = 0.0125 #kappa step per trainpoint
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
    def __init__(self, net, meta, device):
        super().__init__(net, meta, device)
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

    def step(self, labels, images, momentum = 0.0, weight_decay = 0.0, sample_labels=None, sample_images=None):

        net = self.net
        meta = self.meta
        if self.dropout_mode and net.training:
            raise ValueError("For dropout_mode == True net.training must be False")
        logging.info("##Tmp: Calculating labels_to_softhot")
        pp = labels_to_softhot(labels, meta)
        pp_sample = None if sample_labels is None else labels_to_softhot(sample_labels, meta)
        logging.info("##Saving theta")
        self.paramProcessor.save_theta(net)
        if momentum > 0.0 and self.paramProcessor.is_delta_empty() == False:
            logging.info("##Setting grad theta")
            self.paramProcessor.set_theta(net, momentum, 0., 0., weight_decay)

        logging.info("##Calculating params-delta")
        net.zero_grad()
        logits = self.do_forward(images, self.dropout_mode) ## new gradient with dropout is generated here (1*)
        logging.info("##Calculating criterion")
        loss, kappa_avg, kappa_min = self.calc_criterion(logits, labels, pp)
        logging.info("##Criterion calculated")
        norm2_squared_latest = self.paramProcessor.calc_autograd(net, loss, self.do_calc_grad_norm2)
        logging.info("##Autograd calculated")
        with torch.no_grad():
            if self.dropout_mode:
                logits = self.do_forward(images, False) ## all qqxx calculated with dropout off
            #Eta-calculation
            qq0 = self.softmax(logits) #q(t)
            qq0_sample = None if sample_images is None else self.softmax(self.do_forward(sample_images, False))
            #Sample 
            logging.info("##Tmp: Before loss initial")
            #eta_curr = self.eta1 #small step-size
            self.paramProcessor.set_theta(net, momentum, self.eta1, 1.0, weight_decay) #small step
            #logits1 = self.do_forward(images, False)
            qq1 = self.softmax(self.do_forward(images, False))
            delta_pq, delta_qq1 = pp-qq0, qq1-qq0

            logging.info("##Tmp: Before eta_analytic_n2")
            norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro')
            eta2_raw, cos_phi = self.eta_analytic_n2_onehot(self.eta1, delta_pq, delta_qq1, norm_pq, norm_qq1)
            eta2 = eta2_raw * self.alpha
            if self.do_logging:
                logging.info("##alpha={}, eta2={}".format(self.alpha, eta2))
            logging.info("##Tmp: After eta_analytic_n2")
            ck1_armiho = None
            if self.do_calc_armiho:
                loss_initial = self.crossentropy_avg(pp, qq0)
                loss_next = self.crossentropy_avg(pp, qq1)
                diff_initial = -(torch.sum((pp/qq0)*(delta_qq1)))/pp.shape[0]
                ck1_armiho = (loss_next - loss_initial)/(diff_initial+self.epsilon)
                #ck1_wolf = (diff_next)/(diff_initial+self.epsilon)
                if self.do_logging:
                    diff_next = -(torch.sum((pp/qq1)*(delta_qq1)))/pp.shape[0]
                    eta_ratio = eta2/(self.eta1 + self.epsilon)
                    logging.info("##--==On step 1: eta_curr={}, eta_next={}, ratio={}, loss_initial={}, loss_next={}, diff_initial={}, diff_next={}, ck1_armiho={} ==--"\
                                .format(self.eta1, eta2, eta_ratio, loss_initial, loss_next, diff_initial, diff_next, ck1_armiho))
            eta_curr = eta2
            self.paramProcessor.set_theta(net, momentum, eta_curr, 1.0, weight_decay) #1st step
            logging.info("##Tmp:1st step finish")
            cos_phi_sample, ratio_sample = None, None
            if (sample_images is not None):
                qq2_sample = self.softmax(self.do_forward(sample_images, False))
                #logging.info("##Dims: pp_sample={},qq0_sample={}, qq2_sample={}".format(pp_sample.size(), qq0_sample.size(), qq2_sample.size()))
                delta_pq_sample, delta_qq2_sample = pp_sample-qq0_sample, qq2_sample-qq0_sample
                norm_pq_sample, norm_qq2_sample = norm(delta_pq_sample, ord='fro'), norm(delta_qq2_sample, ord='fro')
                cos_phi_sample = torch.sum(delta_pq_sample*delta_qq2_sample)/(norm_pq_sample*norm_qq2_sample + self.epsilon)
                ratio_sample = norm_pq_sample*cos_phi_sample/norm_qq2_sample
                if self.do_logging:
                    logging.info("##--==Sample: norm_pq={}, norm_qq2={}, cos_phi={}, norm_pq*cos_phi/norm_qq2={}"\
                                 .format(norm_pq_sample, norm_qq2_sample, cos_phi_sample, norm_pq_sample*cos_phi_sample/norm_qq2_sample))

            norm2_squared_accumulated = self.paramProcessor.save_delta_current(momentum, eta_curr, 1.0, weight_decay, self.do_calc_grad_norm2)
            logging.info("##Tmp:finish")
            return StepResult( eta_curr, ck1_armiho, norm_pq, norm_qq1, cos_phi, kappa_avg, kappa_min, self.alpha\
                              , norm2_squared_latest, norm2_squared_accumulated, cos_phi_sample, ratio_sample)

class NetLineStepProcessorMSE(NetLineStepProcessorAbstract):
    def __init__(self, net, criterion, meta, device, lbd_dict=None):
        super().__init__(net, meta, device, lbd_dict) 
        self.criterion = criterion if criterion is not None else nn.MSELoss()

    """
    step_params: check_eta2=True/False (default False); set_eta2=True/False (default False)
    """
    def step(self, labels, images, momentum = 0.0, step_params = None):
        net = self.net
        meta = self.meta
        xx, yy = images, labels
        self.paramProcessor.save_theta(net)

        #if momentum > 0.0 and nesterov == True and self.paramProcessor.is_delta_empty() == False:
        #    self.paramProcessor.set_theta(net, momentum, 0., 0.)

        logging.info("##Calculating params-delta")
        net.zero_grad()
        zz = self.do_forward(xx, 'toss') ## new dropout is generated here (1*)
        loss = self.criterion(zz, yy)
        self.paramProcessor.calc_autograd(net, loss)
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
