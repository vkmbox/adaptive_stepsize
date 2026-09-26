import torch
from torch import nn
from torch.linalg import norm
import torch.nn.functional as F

import math
import logging

from common.optim.util import MetaData
from common.optim.optimiser import NetOptimizer


#TODO: support multi param_groups in model
#TODO: support foreach mode
class NetLine(NetOptimizer):
    def __init__(
        self,
        model: nn.Module,
        meta: MetaData,
        lr1: float = 1e-5,
        momentum: float = 0,
        weight_decay: float = 0,
        maximize: bool = False,
        foreach=True,
        lookahead_steps=5,
        lookahead_alpha=1.0
    ) -> None:
        if lr1 < 0.0:
            raise ValueError(f"Invalid learning rate: {lr1}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if lookahead_steps < 0:
            raise ValueError(f"Invalid lookahead_steps value: {lookahead_steps}")
        if lookahead_alpha < 0.0 or 1.0 < lookahead_alpha:
            raise ValueError(f"Invalid lookahead_alpha value: {lookahead_alpha}")

        self.model = model
        self.meta = meta
        self.beta_min = torch.tensor(1e-20).to(meta.device)

        self.do_shorten_lr_for_momentum = True #If momentum > 0, shorten lr by theoretical ratio |g|/|v|
        self.alpha_momentum = 1.0

        self._arctan_coeff = 4.0
        self._one = torch.tensor(1.0).to(meta.device)
        self._eye = torch.eye(meta.output_dim, dtype=torch.float).to(meta.device)

        defaults = {
            "lr1": lr1,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "foreach": foreach
        }
        super().__init__(model.parameters(), defaults, lookahead_steps, lookahead_alpha)
        #if len(self.param_groups) != 1:
        #    raise ValueError("Model must have a single param_group")

        self.init_vars()

    @torch.no_grad()
    def _batch_step(self, images, labels, logitsG, eta_target, fixed_step):
        """Perform a single optimization step.
        """

        logits0 = logitsG
        #Small step
        grads_final = self._step1()

        #Large step eta calculation
        eta1 = self.param_groups[0]["lr1"]
        pp = F.one_hot(labels, self.meta.output_dim)
        qq0 = F.softmax(logits0, dim=1)
        logits1 = self.model.forward(images)
        qq1 = F.softmax(logits1, dim=1)
        delta_pq, delta_q1q = pp-qq0, qq1-qq0
        norm_qq1 = norm(delta_q1q, ord='fro')

        pt_pq_scalar = torch.sum(delta_pq*delta_q1q, dim=1)
        pt_pq_norm = torch.sum(delta_pq*delta_pq, dim=1) ** .5
        pt_q1q_norm = torch.sum(delta_q1q*delta_q1q, dim=1) ** .5
        pt_cos = pt_pq_scalar/(pt_pq_norm*pt_q1q_norm)
        pt_mask = torch.where(pt_cos > 0.25, 1.0, 0.0)
        eta2_raw_nl = torch.sum(pt_mask*pt_pq_norm*pt_q1q_norm*pt_cos*eta1/torch.maximum(norm_qq1**2, self.beta_min))

        eta2_orig_pre = eta2_raw_nl
        eta2_orig, eta2_orig_avg = self._calc_eta_averaging(eta2_orig_pre)
        self.alpha_nomomentum = eta_target/(eta2_orig_avg*self.alpha_momentum)
        alpha_full = self.alpha_nomomentum*self.alpha_momentum
        if (fixed_step):
            eta2 = eta2_pre = self._one * eta_target
        else:
            eta2_pre = eta2_orig_pre * alpha_full
            eta2 = eta2_orig * alpha_full

        logging.debug(f"##net-line: alpha_nomomentum={self.alpha_nomomentum}, alpha_momentum={self.alpha_momentum}, \
                      eta1={eta1}, eta2_pre={eta2_pre}, eta2={eta2}")

        #Large step
        self._step2(grads_final, eta1, eta2)
        return self._step_results(eta2, eta2_pre, alpha_full, self.alpha_nomomentum, qq1)

    def init_vars(self):
        momentum = self.param_groups[0]['momentum']
        self.alpha_momentum = math.sqrt(1-momentum**2) \
            if momentum > 0.0 and self.do_shorten_lr_for_momentum else 1.0

        self._init_eta_averaging()
