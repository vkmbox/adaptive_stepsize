import torch
from torch import Tensor, nn
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
    
def eta(eta_test, delta_pq, delta_qq, norm_pq, norm_qq, epsilon, beta_min, do_logging):
    #with torch.no_grad():
    cos_phi = torch.sum(delta_pq*delta_qq)/(norm_pq*norm_qq + epsilon)
    eta_next = norm_pq*cos_phi*eta_test/torch.maximum(norm_qq, beta_min)
    if do_logging:
        logging.info("##Snl: cos(pp^qq)={}, norm_pq={}, norm_qq={}, eta_test={}, eta_raw={}, beta_min={}"\
                    .format(cos_phi, norm_pq, norm_qq, eta_test, eta_next, beta_min))
    return eta_next, cos_phi

class StepResult:
    def __init__(self, eta, pq_norm=0.0, qq_norm=0.0, cos_phi=0.0, alpha = None\
                 , grad_norm2_squared=None, accum_norm2_squared=None, regression_beta = None, alpha_drift = None):
        self.eta = eta
        self.pq_norm = pq_norm
        self.qq_norm = qq_norm
        self.cos_phi = cos_phi
        self.alpha = alpha
        self.grad_norm2_squared = grad_norm2_squared
        self.accum_norm2_squared = accum_norm2_squared
        self.regression_beta = regression_beta
        self.alpha_drift = alpha_drift

#Epsilon-greedy bandit version
class NetLineStepLR:

    #Values for lr, momentum and weight_decay are set externally in optimiser
    def __init__(self, net, optimizer, meta, foreach=False, loss_fn=None):
        self.net = net
        self.optimizer = optimizer
        self.meta = meta
        self.foreach = foreach
        self.loss_fn=loss_fn if loss_fn is not None else nn.CrossEntropyLoss(reduction='mean')

        self.eta1 = optimizer.param_groups[0]['lr'] if optimizer is not None else .0 #1st step eta-size
        self.alpha_epoch = 0.9 #eta multiplier
        self.beta_min = torch.tensor(0.00001).to(meta.device) #min for eta denom for the eta-calculation stability
        self.epsilon = 1e-9

        self.dropout_mode = False #Set true if the net uses dropout layers
        self.do_logging = False #Is additional params logging performed or not, the logging may affect performance
        self.do_calc_grad_norm2 = False #Is norm2 squared of gradient calculated or not, the calculation may affect performance
        self.do_shorten_lr_for_momentum = False #If momentum > 0, shorten lr by theoretical ratio |g|/|v|

        self.alpha_drift_check = True
        self.regression_queue_size = 100
        self.alpha_drift_min = torch.tensor(0.25, device=meta.device)
        self.alpha_drift_max = torch.tensor(1.25, device=meta.device)
        self.alpha_drift_init = torch.tensor(0.75, device=meta.device)
        self.pid_coeff_linear = 2e2
        self.pid_coeff_integral = 2.0
        self.pid_coeff_diff = 1.0
        self.pid_base = torch.tensor(0.0, device=meta.device)
        #self.coeff_delta_asc = 1.0
        #self.coeff_delta_desc = 1.0
        self.delta_zero = False

        self._alpha_drift = None #latest control signal u(t) value
        self._control_delta = None #latest control delta e(t) value
        self._control_delta_prev = None #latest control delta e(t-1) value

        self._regression_queue = None
        self._regression_multipliers = None
        self._regression_queue_pos = 0
        self._regression_queue_pos_cyclic = False

        self._zero_tenzor = torch.tensor(0.0, device=meta.device)

    #Regression drift-relared part
    def init_regression(self):
        meta = self.meta
        if self.alpha_drift_check:
            self._regression_queue_pos = 0
            self._regression_queue_pos_cyclic = False
            self._regression_queue = \
                torch.zeros((self.regression_queue_size,), dtype=torch.float, device=meta.device)
            self._regression_multipliers = torch.arange(1, self.regression_queue_size + 1, 1).to(meta.device)

    def calc_regression_beta(self, eta):
        self._regression_queue[self._regression_queue_pos] = eta
        #pos shift
        self._regression_queue_pos += 1
        if self._regression_queue_pos >= self.regression_queue_size:
            self._regression_queue_pos_cyclic = True
            self._regression_queue_pos = self._regression_queue_pos - self.regression_queue_size

        shift = self._regression_queue_pos if self._regression_queue_pos_cyclic else 0
        sum1 = torch.sum(torch.roll(self._regression_multipliers, shift)*self._regression_queue)
        sum2 = torch.sum(self._regression_queue)
        return 6*(2*sum1-(self.regression_queue_size+1)*sum2)/(self.regression_queue_size*(self.regression_queue_size**2-1))

    #PID-relared part
    def get_control_delta(self, beta):
        if self.delta_zero:
            return self._zero_tenzor
        return self.pid_base.sub(beta)

    def select_alpha_drift(self, beta):
        control_delta_new = self.get_control_delta(beta)
        if (self._alpha_drift is None):
            self._alpha_drift = self.alpha_drift_init
            self._control_delta = self._zero_tenzor
            self._control_delta_prev = self._zero_tenzor

        '''
        alpha_drift_new = self._alpha_drift + self.pid_coeff_integral * control_delta_new
        alpha_drift_new = alpha_drift_new + self.pid_coeff_linear * (control_delta_new - self._control_delta)
        alpha_drift_new = alpha_drift_new + self.pid_coeff_diff * (control_delta_new - 2*self._control_delta + self._control_delta_prev)
        '''
        delta = control_delta_new.sub(self._control_delta)
        delta_prev = self._control_delta.sub(self._control_delta_prev)
        alpha_drift_new = self._alpha_drift.add(control_delta_new, alpha=self.pid_coeff_integral)
        alpha_drift_new.add_(delta, alpha=self.pid_coeff_linear)
        alpha_drift_new.add_(delta.sub(delta_prev), alpha=self.pid_coeff_diff)

        self._control_delta_prev = self._control_delta
        self._control_delta = control_delta_new
        self._alpha_drift = \
            torch.minimum(torch.maximum(alpha_drift_new, self.alpha_drift_min), self.alpha_drift_max)
        return self._alpha_drift

    def step(self, labels, images):
        net = self.net
        optimizer = self.optimizer

        if self.dropout_mode and net.training:
            raise ValueError("For dropout_mode == True net.training must be False")
        logging.info("##Snl: Step start calculating logits and qq0")
        net.zero_grad()
        logitsG = snl_forward(net, images, self.dropout_mode) ## new gradient with dropout is generated here (1*)
        logging.info("##Snl: calculating criterion")
        loss = self.loss_fn.forward(logitsG, labels)
        logging.info("##Snl: performing small step")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits0 = (logitsG if self.dropout_mode == False else snl_forward(net, images, False)) #.detach().clone()
            return self.internal_step(labels, images, logits0)

    #@_use_grad_for_differentiable
    def internal_step(self, labels, images, logits0):
        net = self.net
        meta = self.meta
        optimizer = self.optimizer

        momentum = optimizer.param_groups[0]['momentum']
        alpha_momentum = math.sqrt(1-momentum**2) if momentum > 0.0 and self.do_shorten_lr_for_momentum else 1.0

        logging.info("##Snl: , calculating pp")
        pp = F.one_hot(labels, meta.output_dim) #.to(meta.device)
        qq0 = F.softmax(logits0, dim=1) + self.epsilon ## all qqxx calculated with dropout off
        logging.info("##Snl: calculating learning rate")
        qq1 = F.softmax(snl_forward(net, images, False), dim=1) + self.epsilon #.detach().clone()
        delta_pq, delta_qq1 = pp-qq0, qq1-qq0

        logging.info("##Snl: calculating eta_analytic_n2")
        norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro') #math.sqrt((delta_pq**2).sum().item()), math.sqrt((delta_qq1**2).sum().item()) #
        eta2_raw, cos_phi = eta(self.eta1, delta_pq, delta_qq1, norm_pq, norm_qq1, self.epsilon, self.beta_min, self.do_logging)

        alpha_drift, regression_beta = 1.0, 0.0
        if self.alpha_drift_check:
            regression_beta = self.calc_regression_beta(eta2_raw)
            if self.do_logging:
                logging.info("##Snl: regression_beta={}".format(regression_beta))
            alpha_drift = self.select_alpha_drift(regression_beta)

        eta2 = eta2_raw*self.alpha_epoch*alpha_momentum*alpha_drift
        if self.do_logging:
            logging.info("##Snl: alpha_epoch={}, alpha_momentum={}, eta2={}".format(self.alpha_epoch, alpha_momentum, eta2))
        logging.info("##Snl: shifting params to the rest of step")
        eta2_shift = eta2.add(-self.eta1)
        grad_norm2_squared, buffer_norm2_squared = 0.0, 0.0

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
                    if self.do_calc_grad_norm2:
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
                if self.do_calc_grad_norm2:
                    for grad_ in torch._foreach_pow(grads, 2.0):
                        grad_norm2_squared += grad_.sum() #.item()
                    for momentum_ in torch._foreach_pow(momentum_buffer_list, 2.0):
                        buffer_norm2_squared += momentum_.sum() #.item()

        logging.info("####Snl: step finish, returning step_result")
        return StepResult( eta2, norm_pq, norm_qq1, cos_phi, self.alpha_epoch*alpha_momentum*alpha_drift,\
                            grad_norm2_squared, buffer_norm2_squared, regression_beta, alpha_drift)
