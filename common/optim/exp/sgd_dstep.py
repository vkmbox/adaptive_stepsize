import torch
from torch import Tensor
from torch.optim import SGD
from typing import List, Optional
#from torch.linalg import norm
import torch.nn.functional as F
from torch.optim.optimizer import _use_grad_for_differentiable
from torch.optim.sgd import sgd

import math

class sgd_dstep(SGD):
    def __init__(
        self,
        params,
        eta1: float = 1e-5,
        momentum: float = .0,
        #dampening: float = 0,
        weight_decay: float = .0,
        #nesterov=False,
        *,
        #maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        super().__init__(params, lr=eta1, momentum=momentum, dampening = 0, weight_decay=weight_decay,\
                         nesterov=False, maximize = False, foreach=foreach, differentiable=differentiable, fused=fused)
        
        self.eta1 = eta1

        self.calc_norm2: bool = False
        self.do_shorten_lr_for_momentum: bool = True
        self.epsilon = 1e-9
        self.beta_min: float = 1e-5

        #set before step
        self.alpha_epoch: float = 1.0
        self.pp: Tensor = None
        self.logits0: Tensor = None


    @_use_grad_for_differentiable
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #loss = None
        #if closure is not None:
        #    with torch.enable_grad():
        #        loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        #shift part
        with torch.no_grad():
            momentum = self.param_groups[0]['momentum']
            alpha_momentum = math.sqrt(1-momentum**2) if momentum > 0.0 and self.do_shorten_lr_for_momentum else 1.0

            #pp = F.one_hot(self.labels, self.meta.output_dim).to(self.meta.device)
            qq0 = F.softmax(self.logits0, dim=1) + self.epsilon ## all qqxx calculated with dropout off
            #qq1 = F.softmax(snl_forward(net, images, False), dim=1) + self.epsilon
            qq1 = F.softmax(closure(), dim=1) + self.epsilon
            delta_pq, delta_qq1 = self.pp-qq0, qq1-qq0

            norm_pq, norm_qq1 = math.sqrt((delta_pq**2).sum().item()), math.sqrt((delta_qq1**2).sum().item()) #norm(delta_pq, ord='fro'), norm(delta_qq1, ord='fro')
            eta2_raw, cos_phi = self.eta(self.eta1, delta_pq, delta_qq1, norm_pq, norm_qq1)
            eta2 = eta2_raw * self.alpha_epoch*alpha_momentum

            eta2_shift = eta2 - self.eta1
            grad_norm2_squared, buffer_norm2_squared = self.shift(eta2_shift)

        return eta2, norm_pq, norm_qq1, cos_phi, self.alpha_epoch*alpha_momentum, grad_norm2_squared, buffer_norm2_squared

    def eta(self, eta_test, delta_pq, delta_qq, norm_pq, norm_qq):
        #with torch.no_grad():
        cos_phi = torch.sum(delta_pq*delta_qq)/(norm_pq*norm_qq + self.epsilon)
        eta_next = norm_pq*cos_phi*eta_test/max(norm_qq, self.beta_min)
        return eta_next, cos_phi

    #@_use_grad_for_differentiable
    def shift(self, eta_shift: float):
        grad_norm2_squared, buffer_norm2_squared = 0.0, 0.0
        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )
            if group["foreach"] == False:
                for num, param in enumerate(params):
                    grad, momentum_buffer = grads[num], momentum_buffer_list[num]
                    vbuff = grad if group["momentum"] == 0 else momentum_buffer
                    param.add_(vbuff, alpha=-eta_shift)
                    if self.calc_norm2:
                        grad_norm2_squared += (grad**2).sum().item() #Check if .item() fine for performance
                        buffer_norm2_squared += (momentum_buffer**2).sum().item()

            else:
                if group["momentum"] == 0:
                    torch._foreach_add_(params, grads, alpha=-eta_shift)
                else:
                    torch._foreach_add_(params, momentum_buffer_list, alpha=-eta_shift)

                if self.calc_norm2:
                    for grad_ in torch._foreach_pow(grads, 2.0):
                        grad_norm2_squared += grad_.sum().item()
                    for momentum_ in torch._foreach_pow(momentum_buffer_list, 2.0):
                        buffer_norm2_squared += momentum_.sum().item()

        return grad_norm2_squared, buffer_norm2_squared
