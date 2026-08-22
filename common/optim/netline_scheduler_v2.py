import torch
from torch import Tensor
from torch.linalg import norm
import torch.nn.functional as F

import math
import logging
from typing import List, Optional

from common.optim.util import AverageCyclicQueue, cosine_annealing2_lr

#force_trainmode=True/False
def snl_forward(net, images, force_evalmode):
    if force_evalmode == True:
        training = net.training
        net.train(False)
        logging.debug("##net-line: --==Explicit train forward==--")
        logits = net.forward(images)
        net.train(training)
        return logits
    else:
        return net.forward(images)

def eta(eta_test, delta_pq, delta_qq, norm_pq, norm_qq, epsilon, beta_min, verbose):
    cos_phi = torch.sum(delta_pq*delta_qq)/torch.maximum(norm_pq*norm_qq, epsilon)
    eta_next = norm_pq*cos_phi*eta_test/torch.maximum(norm_qq, beta_min)
    if verbose:
        logging.debug("##net-line: cos(pp^qq)={}, norm_pq={}, norm_qq={}, eta_test={}, eta_raw={}, beta_min={}"\
                    .format(cos_phi, norm_pq, norm_qq, eta_test, eta_next, beta_min))
    return eta_next, cos_phi

class NetLineStepLR:

    #Values for lr, momentum and weight_decay are set externally in optimiser
    def __init__(self, net, optimizer, meta, foreach=True, la_steps=5, la_alpha=1.0):
        self.net = net
        self.optimizer = optimizer
        self.meta = meta
        self.foreach = foreach
        self._eye = torch.eye(meta.output_dim, dtype=torch.float).to(meta.device)
        self._zero = torch.tensor(0.0).to(meta.device)
        self._one = torch.tensor(1.0).to(meta.device)

        self.beta_min = torch.tensor(1e-10).to(meta.device) #min for eta denom for the eta-calculation stability
        self.epsilon = torch.tensor(1e-20).to(meta.device)
        self.y_part = 0.0

        self.dropout_mode = False #Set true if the net uses dropout layers
        self.verbose = False #Is additional params logging performed or not, the logging may affect performance
        self.do_shorten_lr_for_momentum = True #If momentum > 0, shorten lr by theoretical ratio |g|/|v|
        self.alpha_momentum = 1.0

        self.lr_averaging_check_up = 1.0
        self.lr_averaging_check_down = 1.0
        self.lr_averaging_queue_size = 150
        self._lr_averaging_queue = None

        self.la_alpha = la_alpha
        self._la_step = 0  # counter for inner optimizer
        self._total_la_steps = la_steps
        self.la_state: List[Tensor] = []
        self.la_backup: List[Tensor] = []

        self.alpha_nomomentum = 0.75
        self.alpha_nomomentum_max = self._one *2.0

        self._flag_check_no_backstep = False

        self.lr_max = 2e-2
        self.eta_target = 2e-2
        self.eta_target_min = 1e-5

        self.epochs_per_experiment = 50
        self.epochs_warmup = 3
        #self.epochs_sampling = -1
        #self.epochs_wide = -1
        #self.epochs_middle = -1

        #self.averaging_wide_up = 1.0
        #self.averaging_wide_down = 1.0
        #self.averaging_middle_up = 0.1
        #self.averaging_middle_down = 0.1
        self._arctan_coeff = 4.0
        #self._eta_target_as_eta1 = False #deprecated

        self._epoch = 0

    def init_params(self):
        momentum = self.optimizer.param_groups[0]['momentum']
        self.alpha_momentum = math.sqrt(1-momentum**2) \
            if momentum > 0.0 and self.do_shorten_lr_for_momentum else 1.0

        self.step(epoch = -1)

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

    def init_eta_averaging(self):
        self._lr_averaging_queue = AverageCyclicQueue(queue_size = self.lr_averaging_queue_size, fill_value = 0.02, device = self.meta.device)

    def calc_eta_averaging(self, eta):
        self._lr_averaging_queue.put_value(eta)

        #if (self.lr_averaging_check_up >= 1.0 and self.lr_averaging_check_down >= 1.0):
        #    return eta

        if not self._lr_averaging_queue._pos_cyclic:
            return eta
        else:
            eta_avg = self._lr_averaging_queue.get_avg()
            if (self.lr_averaging_check_up <= 0.0 and self.lr_averaging_check_down <= 0.0):
                return eta_avg
            else:
                eta_delta0 = eta - eta_avg
                eta_delta1 = eta_delta0
                if (self.lr_averaging_check_up != 1.0 or self.lr_averaging_check_down != 1.0):
                    eta_delta1 = eta_delta0*torch.where(torch.sign(eta_delta0) == self._one,\
                                                        self.lr_averaging_check_up, self.lr_averaging_check_down)
                eta_delta = torch.arctan(eta_delta1*self._arctan_coeff/eta_avg)*eta_avg/self._arctan_coeff
                return eta_avg + eta_delta

    def step(self, epoch = None):
        """Method to call in every epoch for scheduler/optimiser params adjustment. It is called after the epoch's training loop
        """
        if epoch is not None:
            self._epoch = epoch

        self._epoch += 1

        self.eta_target = cosine_annealing2_lr(self.lr_max, 0.0, 0, self.epochs_per_experiment, self._epoch)
        #if self._eta_target_as_eta1:
        #    self.optimizer.param_groups[0]['lr'] = max(self.eta_target, self.eta_target_min)

    def batch_step(self, x, y, y_pred, **kwargs):
        """Method to call in every minibatch together with step call in every epoch. Loss forward-backward performed externally
        """

        images, labels, logitsG = x, y, y_pred
        fixed_step = self._epoch < self.epochs_warmup or self._lr_averaging_queue._pos_cyclic == False
        self.optimizer.step()

        with torch.no_grad():
            return self._step_nl(labels, images, logitsG, eta_target = self.eta_target, fixed_step = fixed_step)

    def _step_nl(self, labels, images, logitsG, eta_target, fixed_step):
        net = self.net
        meta = self.meta
        optimizer = self.optimizer
        eta1 = optimizer.param_groups[0]['lr'] #1st step eta-size

        logits0 = (logitsG if self.dropout_mode == False else snl_forward(net, images, force_evalmode=self.dropout_mode))

        logging.debug("##net-line: calculating pp")
        pp = F.one_hot(labels, meta.output_dim)
        qq0 = F.softmax(logits0, dim=1) ## all qqxx calculated with dropout-off/eval-mode
        logging.debug("##net-line: calculating learning rate")
        logits1 = snl_forward(net, images, force_evalmode=self.dropout_mode)
        qq1 = F.softmax(logits1, dim=1) #0-point, 1-neuron?
        delta_pq, delta_q1q = pp-qq0, qq1-qq0

        logging.debug("##net-line: calculating eta_preactivation")
        eta2_raw_y = 0.0
        if self.y_part > 0.0:
            dz = (logits1-logits0)/eta1
            qqq = qq0[:,:,None]*(self._eye[None,:,:]-qq0[:,None,:])
            eta2_raw_y = torch.squeeze(torch.sum(delta_pq*dz)/torch.sum(dz[:,:,None]*qqq*dz[:,None,:]))

        logging.debug("##net-line: calculating eta_analytic_n2")
        norm_pq, norm_qq1 = norm(delta_pq, ord='fro'), norm(delta_q1q, ord='fro')
        eta2_raw, cos_phi = eta(eta1, delta_pq, delta_q1q, norm_pq, norm_qq1, self.epsilon, self.beta_min, self.verbose)

        eta2_orig_pre = ((1.0 - self.y_part)*eta2_raw + self.y_part * eta2_raw_y)
        eta2_orig = self.calc_eta_averaging(eta2_orig_pre)
        eta2_orig_avg = self._lr_averaging_queue.get_avg()
        self.alpha_nomomentum = torch.minimum(eta_target/(eta2_orig_avg*self.alpha_momentum), self.alpha_nomomentum_max)
        alpha_full = self.alpha_nomomentum*self.alpha_momentum
        if (fixed_step):
            eta2 = eta2_pre = self._one * eta_target
        else:
            eta2_pre = eta2_orig_pre * alpha_full
            eta2 = eta2_orig * alpha_full

        if self.verbose:
            logging.debug("##net-line: alpha_epoch={}, alpha_momentum={}, eta2_pre={}, eta2={}".\
                         format(self.alpha_epoch, self.alpha_momentum, eta2_pre, eta2))
        logging.debug("##net-line: shifting params to the rest of step")

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

                    if (not self._flag_check_no_backstep or eta2 > self._zero):
                        eta2_shift = eta2.add(-eta1)
                        buffer_x_shift = None
                        if group["momentum"] == 0:
                            buffer_x_shift = grad.mul(-eta2_shift)
                        else:
                            buffer_x_shift = momentum_buffer.mul(-eta2_shift)
                        param.add_(buffer_x_shift)

                    if do_lookahead:
                        # Lookahead and cache the current optimizer parameters
                        param_state = self.la_state[num]
                        if self.la_alpha != 1.0:
                            param.mul_(self.la_alpha).add_(param_state, alpha=1.0 - self.la_alpha)
                        param_state.copy_(param)

            else:
                if (not self._flag_check_no_backstep or eta2 > self._zero):
                    eta2_shift = eta2.add(-eta1)
                    buffers_x_shift = None
                    if group["momentum"] == 0:
                        buffers_x_shift = torch._foreach_mul(grads, -eta2_shift)
                    else:
                        buffers_x_shift = torch._foreach_mul(momentum_buffer_list, -eta2_shift)
                    torch._foreach_add_(params, buffers_x_shift)

                if do_lookahead:
                    if self.la_alpha != 1.0:
                        torch._foreach_mul_(params, self.la_alpha)
                        torch._foreach_add_(params, self.la_state, alpha=1.0 - self.la_alpha)
                    torch._foreach_copy_(self.la_state, params)

        logging.debug("####net-line: step finish, returning step_result")
        return self.step_results(eta2, eta2_pre, norm_pq, norm_qq1, cos_phi, alpha_full, self.alpha_nomomentum, qq0)

    def step_results(self, eta, eta2_pre, pq_norm, qq_norm, cos_phi, alpha, alpha_nomomentum, qq0):
        result = {}
        result['eta'] = eta
        result['eta2_pre'] = eta2_pre
        result['pq_norm'] = pq_norm
        result['qq_norm'] = qq_norm
        result['cos_phi'] = cos_phi
        result['alpha'] = alpha
        result['alpha_nomomentum'] = alpha_nomomentum
        result['qq0'] = qq0

        return result

class MetaData:
    def __init__(self, output_dim = 10, device='cpu'):
        self.device = device
        self.output_dim = output_dim
