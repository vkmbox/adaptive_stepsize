import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.linalg import norm
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, _get_value

import logging

from common.optim.utils import MetaData

class AdamNetLine(Optimizer):

    def __init__(self, model: nn.Module, optimizer: optim.Adam, meta: MetaData, alpha: float = 1.0,\
                  predicted_type: str="log_softmax", eta_max: float = 1e-2):

        #TODO: check optimizer is not capturable or differentiable
        self.optimizer = optimizer
        predicted_type = predicted_type.lower()
        assert predicted_type in ["log_softmax", "logit"]
        self.predicted_type = predicted_type
        self.meta = meta
        self.model = model
        self.alpha = alpha
        self.eta_max = torch.tensor(eta_max).to(meta.device)
        self.eps = torch.tensor(1e-8).to(meta.device) #optimizer.param_groups[0]["eps"]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @torch.no_grad()
    def step(self, x, y, out_predicted):
        batchnorms = [
            module for module in self.model.modules()
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
            and module.training and module.track_running_stats
        ]
        res = None
        for module in batchnorms:
            module.track_running_stats = False
        try:
            res = self._batch_step(x, y, out_predicted)
        finally:
            for module in batchnorms:
                module.track_running_stats = True

        return res

    def _batch_step(self, images, labels, out_predicted0):

        def get_softmax(out_predicted):
            if self.predicted_type == "logit":
                return F.softmax(out_predicted, dim=1)
            elif self.predicted_type == "log_softmax":
                return torch.exp(out_predicted)
            else:
                raise ValueError(f"Unknown predicted type={self.predicted_type}")

        opt = self.optimizer

        #logits0 = logitsG
        qq0 = get_softmax(out_predicted0)

        #Small step
        opt.step()

        #Large step eta calculation
        eta1 = opt.param_groups[0]["lr"]
        pp = F.one_hot(labels, self.meta.output_dim)
        out_predicted1 = self.model.forward(images)
        qq1 = get_softmax(out_predicted1)
        delta_pq, delta_q1q = pp-qq0, qq1-qq0
        norm_q1q = norm(delta_q1q, ord='fro')
        #norm_pq = norm(delta_pq, ord='fro')

        #cos_phi = torch.sum(delta_pq*delta_q1q)/(norm_pq*norm_q1q + self.eps)
        #eta2_raw = norm_pq*cos_phi*eta1/(norm_q1q + self.eps)
        eta2_raw = self.alpha*eta1*torch.sum(delta_pq*delta_q1q)/(norm_q1q**2 + self.eps) #torch.maximum(norm_q1q**2, self.eps)

        '''
        pt_pq_scalar = torch.sum(delta_pq*delta_q1q, dim=1)
        #logging.info(f"##Adam NetLine: pt_pq_scalar={pt_pq_scalar}")
        pt_pq_norm = torch.sum(delta_pq*delta_pq, dim=1) ** .5
        #logging.info(f"##Adam NetLine: pt_pq_norm={pt_pq_norm}")
        pt_q1q_norm = torch.sum(delta_q1q*delta_q1q, dim=1) ** .5
        #logging.info(f"##Adam NetLine: pt_q1q_norm={pt_q1q_norm}")
        pt_cos = pt_pq_scalar/(pt_pq_norm*pt_q1q_norm + self.eps)
        #logging.info(f"##Adam NetLine: pt_cos={pt_cos}")
        pt_mask = torch.where(pt_cos >= -1.0, 1.0, 0.0)
        #logging.info(f"##Adam NetLine: pt_mask={pt_mask}")
        eta2_raw = torch.sum(pt_mask*pt_pq_norm*pt_q1q_norm*pt_cos*eta1/torch.maximum(norm_q1q**2, self.eps))
        '''
        eta2 = torch.minimum(eta2_raw, self.eta_max)
        logging.info(f"##Adam NetLine: eta2_raw={eta2_raw}, eta2={eta2}")

        #additional params shift
        eta2_shift = eta2.add(-eta1)
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]
            amsgrad = group["amsgrad"]
            eps=group["eps"]

            opt._init_group(
                group,
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            for num, param in enumerate(params):
                exp_avg = exp_avgs[num]
                exp_avg_sq = exp_avg_sqs[num]
                step_t = state_steps[num]

                step = _get_value(step_t)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = eta2_shift / bias_correction1

                bias_correction2_sqrt = bias_correction2**0.5

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[num], exp_avg_sq, out=max_exp_avg_sqs[num])

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[num].sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)

        return self._step_results(eta2, qq1)

    def _step_results(self, eta, qq1):
        result = {}
        result['eta'] = eta
        result['qq1'] = qq1
        return result
