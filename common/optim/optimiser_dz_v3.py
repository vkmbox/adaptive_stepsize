import torch
from torch import Tensor, nn
import torch.nn.functional as F

import math
import logging

from common.optim.util import MetaData
from common.optim.optimiser import NetOptimizer


#TODO: support multi param_groups in model
#TODO: support foreach mode
class NetDz(NetOptimizer):
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
        if len(self.param_groups) != 1:
            raise ValueError("Model must have a single param_group")

        self.init_vars()

    @torch.no_grad()
    def _batch_step(self, images, labels, logitsG, eta_target, fixed_step):
        """Perform a single optimization step.
        """

        result = None

        #for group in self.param_groups:
        group = self.param_groups[0]
        params: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_buffer_list: list[Tensor | None] = []

        has_sparse_grad = self._init_group(
            group, params, grads, momentum_buffer_list
        )

        result = self._single_tensor_netline(
            images=images,
            labels=labels,
            logits0=logitsG,
            params=params,
            grads=grads,
            momentum_buffer_list=momentum_buffer_list,
            weight_decay=group["weight_decay"],
            momentum=group["momentum"],
            eta1=group["lr1"],
            maximize=group["maximize"],
            #foreach=group["foreach"],
            eta_target = eta_target,
            fixed_step = fixed_step
        )

        if group["momentum"] != 0:
            # update momentum_buffers in state
            for param, momentum_buffer in zip(
                params, momentum_buffer_list, strict=True
            ):
                stat = self.state[param]
                stat["momentum_buffer"] = momentum_buffer

        return result

    def _single_tensor_netline(
        self,
        images: Tensor,
        labels: Tensor,
        logits0: Tensor,
        params: list[Tensor],
        grads: list[Tensor],
        momentum_buffer_list: list[Tensor | None],
        weight_decay: float,
        momentum: float,
        eta1: float,
        maximize: bool,
        eta_target: float,
        fixed_step: bool
    ) -> None:
        meta = self.meta
        grads_final = {}
        #Small step
        for num, param in enumerate(params):
            grad = grads[num] if not maximize else -grads[num]

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            grads_final[num] = grad.detach().clone()

            param_shift1 = grads_final[num].mul(eta1)
            if momentum != 0:
                buf = momentum_buffer_list[num]
                if buf is None:
                    momentum_buffer_list[num] = param_shift1
                else:
                    buf.add_(param_shift1)

            param.sub_(param_shift1)

        #Large step eta calculation
        pp = F.one_hot(labels, meta.output_dim)
        qq0 = F.softmax(logits0, dim=1)
        logits1 = self.model.forward(images)
        qq1 = F.softmax(logits1, dim=1)
        delta_pq, delta_q1q = pp-qq0, qq1-qq0

        pt_pq_scalar = torch.sum(delta_pq*delta_q1q, dim=1)
        pt_pq_norm = torch.sum(delta_pq*delta_pq, dim=1) ** .5
        pt_q1q_norm = torch.sum(delta_q1q*delta_q1q, dim=1) ** .5
        pt_cos = pt_pq_scalar/(pt_pq_norm*pt_q1q_norm)
        pt_mask = torch.where(pt_cos > 0.25, 1.0, 0.0)

        dz = (logits1-logits0)/eta1
        qqq = pt_mask[:,None,None]*qq0[:,:,None]*(self._eye[None,:,:]-qq0[:,None,:])
        eta2_raw_y = torch.squeeze(torch.sum(pt_mask[:,None]*delta_pq*dz)/torch.sum(dz[:,:,None]*qqq*dz[:,None,:]))

        eta2_orig_pre = eta2_raw_y
        eta2_orig, eta2_orig_avg = self._calc_eta_averaging(eta2_orig_pre)
        self.alpha_nomomentum = eta_target/(eta2_orig_avg*self.alpha_momentum)
        alpha_full = self.alpha_nomomentum*self.alpha_momentum
        if (fixed_step):
            eta2 = eta2_pre = self._one * eta_target
        else:
            eta2_pre = eta2_orig_pre * alpha_full
            eta2 = eta2_orig * alpha_full

        logging.debug(f"##net-line: alpha_nomomentum={self.alpha_nomomentum}, alpha_momentum={self.alpha_momentum}, eta1={eta1}, eta2_pre={eta2_pre}, eta2={eta2}")

        #Large step
        eta2_shift = eta2.add(-eta1)
        for num, param in enumerate(params):
            param_shift2 = grads_final[num].mul(eta2_shift)
            param.sub_(param_shift2)
            if momentum != 0:
                momentum_buffer_list[num].add_(param_shift2)

        return self._step_results(eta2, eta2_pre, alpha_full, self.alpha_nomomentum, qq1)

    def init_vars(self):
        momentum = self.param_groups[0]['momentum']
        self.alpha_momentum = math.sqrt(1-momentum**2) \
            if momentum > 0.0 and self.do_shorten_lr_for_momentum else 1.0

        self._init_eta_averaging()
