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
    cos_phi = torch.sum(delta_pq*delta_qq)/torch.maximum(norm_pq*norm_qq, epsilon)
    eta_next = norm_pq*cos_phi*eta_test/torch.maximum(norm_qq, beta_min)
    if do_logging:
        logging.info("##Snl: cos(pp^qq)={}, norm_pq={}, norm_qq={}, eta_test={}, eta_raw={}, beta_min={}"\
                    .format(cos_phi, norm_pq, norm_qq, eta_test, eta_next, beta_min))
    return eta_next, cos_phi

class StepResult:
    def __init__(self, eta, eta2_pre=0.0, pq_norm=0.0, qq_norm=0.0, cos_phi=0.0, alpha = None, alpha_nomomentum = None\
                 , grad_norm2_squared=None, accum_norm2_squared=None, regression_beta = None, alpha_drift = None, qq0 = None):
        self.eta = eta
        self.eta2_pre = eta2_pre
        self.pq_norm = pq_norm
        self.qq_norm = qq_norm
        self.cos_phi = cos_phi
        self.alpha = alpha
        self.alpha_nomomentum = alpha_nomomentum
        self.grad_norm2_squared = grad_norm2_squared
        self.accum_norm2_squared = accum_norm2_squared
        self.regression_beta = regression_beta
        self.alpha_drift = alpha_drift
        self.qq0 = qq0

class NetLineStepLR:

    #Values for lr, momentum and weight_decay are set externally in optimiser
    def __init__(self, net, optimizer, meta, la_steps=5, la_alpha=0.8, foreach=False, loss_fn=None):
        self.net = net
        self.optimizer = optimizer
        self.meta = meta
        self.foreach = foreach
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(reduction='mean')
        self.loss_fn_kappas = None
        self.loss_kappas = False
        self._eye = torch.eye(meta.output_dim, dtype=torch.float).to(meta.device)

        self.alpha_epoch = 0.9 #eta multiplier
        self.beta_min = torch.tensor(1e-10).to(meta.device) #min for eta denom for the eta-calculation stability
        self.epsilon = torch.tensor(1e-20).to(meta.device)
        self.y_part = 0.0

        self.dropout_mode = False #Set true if the net uses dropout layers
        self.do_logging = False #Is additional params logging performed or not, the logging may affect performance
        self.do_calc_grad_norm2 = False #Is norm2 squared of gradient calculated or not, the calculation may affect performance
        self.do_shorten_lr_for_momentum = False #If momentum > 0, shorten lr by theoretical ratio |g|/|v|
        self.alpha_momentum = math.sqrt(1-optimizer.param_groups[0]['momentum']**2) \
            if optimizer.param_groups[0]['momentum'] > 0.0 and self.do_shorten_lr_for_momentum else 1.0

        self.lr_drift_check = True
        self.lr_drift_size = 500
        self.lr_drift_pos = 5.0
        self.lr_drift_neg = 0.0

        self._zero = torch.tensor(0.0).to(meta.device)
        self.alpha_drift_min = torch.tensor(0.75).to(meta.device)
        self.alpha_drift_max = torch.tensor(1.25).to(meta.device)
        self.alpha_drift_fix = 1.0
        self._lr_drift_queue = None
        self._lr_drift_multipliers = None
        self._lr_drift_queue_pos = 0
        self._lr_drift_queue_pos_cyclic = False

        self.lr_averaging_check = 1.0
        self.lr_averaging_size = 100
        self._lr_averaging_queue_pos = 0
        self._lr_averaging_queue_pos_cyclic = False

        self.la_alpha = la_alpha
        self._la_step = 0  # counter for inner optimizer
        self._total_la_steps = la_steps
        self.la_state: List[Tensor] = []
        self.la_backup: List[Tensor] = []

        self._flag_check_no_backstep = False

    def init_params(self):
        momentum = self.optimizer.param_groups[0]['momentum']
        self.alpha_momentum = math.sqrt(1-momentum**2) \
            if momentum > 0.0 and self.do_shorten_lr_for_momentum else 1.0

    def init_lookahead(self):
        # Cache the current optimizer parameters params.append(p)
        for group in self.optimizer.param_groups:
            for param in group['params']:
                param_state = torch.zeros_like(param)
                param_state.copy_(param)
                self.la_state.append(param_state)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        self.la_backup.clear()
        for group in self.optimizer.param_groups:
            for num, param in enumerate(group['params']):
                backup_params = torch.zeros_like(param.data)
                backup_params.copy_(param.data)
                self.la_backup.append(backup_params)
                param.data.copy_(self.la_state[num])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for num, param in enumerate(group['params']):
                param.data.copy_(self.la_backup[num])
        self.la_backup.clear()

    def init_averaging(self):
        meta = self.meta
        #if self.lr_averaging_check < 1.0:
        self._lr_averaging_queue_pos = 0
        self._lr_averaging_queue_pos_cyclic = False
        self._lr_averaging_queue = \
            torch.full((self.lr_averaging_size,), fill_value=0.02, dtype=torch.float).to(meta.device)

    def calc_averaging(self, eta):
        if self.lr_averaging_check >= 1.0:
            return eta
        self._lr_averaging_queue[self._lr_averaging_queue_pos] = eta
        #pos shift
        self._lr_averaging_queue_pos += 1
        if self._lr_averaging_queue_pos >= self.lr_averaging_size:
            self._lr_averaging_queue_pos_cyclic = True
            self._lr_averaging_queue_pos = 0

        if not self._lr_averaging_queue_pos_cyclic:
            return eta
        else:
            eta_avg = torch.mean(self._lr_averaging_queue)
            if self.lr_averaging_check <= 0.0:
                return eta_avg
            else:
                return eta_avg + (eta - eta_avg)*self.lr_averaging_check

    def init_regression(self):
        meta = self.meta
        if self.lr_drift_check:
            self._lr_drift_queue_pos = 0
            self._lr_drift_queue_pos_cyclic = False
            self._lr_drift_queue = \
                torch.full((self.lr_drift_size,), fill_value=-0.025, dtype=torch.float).to(meta.device)
            self._lr_drift_multipliers = torch.arange(1, self.lr_drift_size + 1, 1).to(meta.device)

    def calc_regression_beta(self, eta):
        self._lr_drift_queue[self._lr_drift_queue_pos] = eta

        #pos shift
        self._lr_drift_queue_pos += 1
        if self._lr_drift_queue_pos >= self.lr_drift_size:
            self._lr_drift_queue_pos_cyclic = True
            self._lr_drift_queue_pos = 0

        shift = self._lr_drift_queue_pos if self._lr_drift_queue_pos_cyclic else 0
        sum1 = torch.sum(torch.roll(self._lr_drift_multipliers, shift)*self._lr_drift_queue)
        sum2 = torch.sum(self._lr_drift_queue)
        return 6*(2*sum1-(self.lr_drift_size+1)*sum2)/(self.lr_drift_size*(self.lr_drift_size**2-1))

    def calc_regression_adjustment(self, beta):
        return 1.0 + torch.where(beta < 0.0, beta*self.lr_drift_neg, beta*self.lr_drift_pos)

    def step(self, labels, images, do_second_step = True):
        net = self.net
        optimizer = self.optimizer

        if self.dropout_mode and net.training:
            raise ValueError("For dropout_mode == True net.training must be False")
        logging.info("##Snl: Step start calculating logits and qq0")
        net.zero_grad()
        logitsG = snl_forward(net, images, self.dropout_mode) ## new gradient with dropout is generated here (1*)
        with torch.no_grad():
            logits0 = (logitsG if self.dropout_mode == False else snl_forward(net, images, False))
        logging.info("##Snl: calculating criterion")

        if self.loss_kappas:
            loss, kappas = self.loss_fn_kappas.forward(logitsG, labels)
        else:
            loss, kappas = self.loss_fn.forward(logitsG, labels), None
        logging.info("##Snl: performing small step")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            return self.internal_step(labels, images, logits0, kappas, do_second_step)

    def internal_step(self, labels, images, logits0, kappas, do_second_step = True):
        net = self.net
        meta = self.meta
        optimizer = self.optimizer
        eta1 = optimizer.param_groups[0]['lr'] #1st step eta-size

        logging.info("##Snl: , calculating pp")
        pp = F.one_hot(labels, meta.output_dim)
        qq0 = F.softmax(logits0, dim=1) ## all qqxx calculated with dropout off
        logging.info("##Snl: calculating learning rate")
        logits1 = snl_forward(net, images, False)
        qq1 = F.softmax(logits1, dim=1) #0-point, 1-neuron?
        delta_pq, delta_qq1 = pp-qq0, qq1-qq0

        logging.info("##Snl: calculating eta_preactivation")
        dz = (logits1-logits0)/eta1
        if kappas is None:
            qqq = qq0[:,:,None]*(self._eye[None,:,:]-qq0[:,None,:])
            eta2_raw_y = torch.squeeze(torch.sum(delta_pq*dz)/torch.sum(dz[:,:,None]*qqq*dz[:,None,:]))
        else:
            qqq = kappas[:,None,None]*qq0[:,:,None]*(self._eye[None,:,:]-qq0[:,None,:])
            eta2_raw_y = torch.squeeze(torch.sum(kappas[:,None]*delta_pq*dz)/torch.sum(dz[:,:,None]*qqq*dz[:,None,:]))

        logging.info("##Snl: calculating eta_analytic_n2")
        norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro')
        eta2_raw, cos_phi = eta(eta1, delta_pq, delta_qq1, norm_pq, norm_qq1, self.epsilon, self.beta_min, self.do_logging)
        alpha_drift, regression_beta = self.alpha_drift_fix, 0.0
        if self.lr_drift_check:
            regression_beta = self.calc_regression_beta(norm_qq1/eta1)
            if self.do_logging:
                logging.info("##Snl: regression_beta={}".format(regression_beta))
            alpha_drift = \
                torch.minimum(torch.maximum(self.calc_regression_adjustment(regression_beta), self.alpha_drift_min), self.alpha_drift_max)

        #eta2_q = eta2_raw*self.alpha_momentum
        #eta2_y = eta2_raw_y*self.alpha_momentum
        eta2_momentum = ((1.0 - self.y_part)*eta2_raw + self.y_part * eta2_raw_y)*self.alpha_momentum
        if (do_second_step):
            alpha_nomomentum=self.alpha_epoch*alpha_drift
            eta2_pre = eta2_momentum*alpha_nomomentum
            eta2 = self.calc_averaging(eta2_pre)
        else:
            eta2 = eta2_pre = optimizer.param_groups[0]['lr']
            alpha_nomomentum = eta2_pre/eta2_momentum

        if self.do_logging:
            logging.info("##Snl: alpha_epoch={}, alpha_momentum={}, eta2_pre={}, eta2={}".format(self.alpha_epoch, self.alpha_momentum, eta2_pre, eta2))
        logging.info("##Snl: shifting params to the rest of step")
        grad_norm2_squared, buffer_norm2_squared = 0.0, 0.0

        do_lookahead = False
        if self.la_alpha < 1.0:
            self._la_step += 1
            if self._la_step >= self._total_la_steps:
                self._la_step, do_lookahead = 0, True

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

                    if do_second_step:
                        eta2_shift = eta2.add(-eta1)
                        if (not self._flag_check_no_backstep or eta2_shift > self._zero):
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

                    if do_lookahead:
                        # Lookahead and cache the current optimizer parameters
                        param_state = self.la_state[num]
                        if self.la_alpha != 1.0:
                            param.mul_(self.la_alpha).add_(param_state, alpha=1.0 - self.la_alpha)
                        param_state.copy_(param)

            else:
                if do_second_step:
                    eta2_shift = eta2.add(-eta1)
                    if (not self._flag_check_no_backstep or eta2_shift > self._zero):
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

                if do_lookahead:
                    if self.la_alpha != 1.0:
                        torch._foreach_mul_(params, self.la_alpha)
                        torch._foreach_add_(params, self.la_state, alpha=1.0 - self.la_alpha)
                    #for num, param in enumerate(params):
                    #    self.la_state[num].copy_(param.data)
                    torch._foreach_copy_(self.la_state, params)

        logging.info("####Snl: step finish, returning step_result")
        return StepResult( eta2, eta2_pre, norm_pq, norm_qq1, cos_phi, self.alpha_epoch*self.alpha_momentum*alpha_drift, alpha_nomomentum,\
                            grad_norm2_squared, buffer_norm2_squared, regression_beta, alpha_drift, qq0)
