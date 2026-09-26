import torch
from torch import Tensor
from torch import nn, optim
from torch.linalg import norm
import torch.nn.functional as F

import math
import logging
from common.optim.util import AverageCyclicQueue, MetaData


#TODO: support multi param_groups in model
#TODO: support foreach mode
class NetDz(optim.Optimizer):
    def __init__(
        self,
        model: nn.Module,
        meta: MetaData,
        lr1: float = 1e-5,
        momentum: float = 0,
        weight_decay: float = 0,
        maximize: bool = False
    ) -> None:
        if lr1 < 0.0:
            raise ValueError(f"Invalid learning rate: {lr1}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.model = model
        self.meta = meta
        self.lr_averaging_queue_size = 50
        self._lr_averaging_queue = None

        self.do_shorten_lr_for_momentum = True #If momentum > 0, shorten lr by theoretical ratio |g|/|v|
        self.alpha_momentum = 1.0

        self._arctan_coeff = 4.0
        self._one = torch.tensor(1.0).to(meta.device)
        self._eye = torch.eye(meta.output_dim, dtype=torch.float).to(meta.device)

        defaults = {
            "lr1": lr1,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "maximize": maximize
        }
        super().__init__(model.parameters(), defaults)
        if len(self.param_groups) != 1:
            raise ValueError("Model must have a single param_group")

        self.init_vars()

    def __setstate__(self, state):
        super().__setstate__(state)
        #for group in self.param_groups:
        group = self.param_groups[0]
        group.setdefault("maximize", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for param in group["params"]:
            params.append(param)
            if param.grad is not None:
                grads.append(param.grad)
                if param.grad.is_sparse:
                    has_sparse_grad = True

            if group["momentum"] != 0:
                state = self.state[param]
                momentum_buffer_list.append(state.get("momentum_buffer"))

        return has_sparse_grad

    @torch.no_grad()
    def batch_prestep(self):

        #for group in self.param_groups:
        group = self.param_groups[0]
        params: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_buffer_list: list[Tensor | None] = []

        self._init_group(
            group, params, grads, momentum_buffer_list
        )

        momentum=group["momentum"]
        if momentum != 0:
            for num, param in enumerate(params):
                buf = momentum_buffer_list[num]
                if buf is not None:
                    buf.mul_(momentum)
                    param.sub_(buf)
                    stat = self.state[param]
                    stat["momentum_buffer"] = buf

    @torch.no_grad()
    def batch_step(self, x, y, y_pred, eta_target, epoch_fixed_step = False, arctan_coeff = None, **kwargs):
        """Method to call in every minibatch together with step call in every epoch. Loss forward-backward performed externally
        """

        images, labels, logitsG = x, y, y_pred

        fixed_step = epoch_fixed_step or self._lr_averaging_queue._pos_cyclic == False
        if arctan_coeff is not None: self._arctan_coeff = arctan_coeff
        return self._batch_step(images, labels, logitsG, eta_target = eta_target, fixed_step = fixed_step)

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

    def _init_eta_averaging(self):
        self._lr_averaging_queue = AverageCyclicQueue(queue_size = self.lr_averaging_queue_size, fill_value = 0.02, device = self.meta.device)

    def _calc_eta_averaging(self, eta):
        self._lr_averaging_queue.put_value(eta)

        if not self._lr_averaging_queue._pos_cyclic:
            return eta, eta
        else:
            eta_avg = self._lr_averaging_queue.get_avg()
            eta_delta0 = (eta - eta_avg)
            eta_delta = torch.arctan(eta_delta0*self._arctan_coeff/eta_avg)*eta_avg/self._arctan_coeff
            return eta_avg + eta_delta, eta_avg

    def _step_results(self, eta, eta2_pre, alpha, alpha_nomomentum, qq1):
        result = {}
        result['eta'] = eta
        result['eta2_pre'] = eta2_pre
        result['alpha'] = alpha
        result['alpha_nomomentum'] = alpha_nomomentum
        result['qq1'] = qq1
        return result
