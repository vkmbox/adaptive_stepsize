import torch
from torch import Tensor
from torch.optim import SGD
from typing import cast, List, Optional, Union
from torch.optim.optimizer import _use_grad_for_differentiable


class SGD_LEAP(SGD):
    """LEAP implementation (Liu et al., CIKM'22, arXiv:2208.11873).

    Adds a per-parameter Gaussian perturbation
    to the learning rate: h ~ N(eta * 1, (eta * sigma)^2 * I)   [paper Eq. 1]

    Note: sigma is RELATIVE to the current lr (paper's formulation), so the
    perturbation automatically scales down as the schedule decays eta.
    """
    
    def __init__(self, *args, sigma=0.1, **kwargs):
        super().__init__(*args, **kwargs, foreach=False, fused=False)
        self.sigma = sigma


    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            sgd_leap(
                params,
                grads,
                momentum_buffer_list,
                sigma=self.sigma,
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

        return loss

def sgd_leap(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    sigma: float,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    # if foreach is None and fused is None:
    #     # why must we be explicit about an if statement for torch.jit.is_scripting here?
    #     # because JIT can't handle Optionals nor fancy conditionals when scripting
    #     if not torch.jit.is_scripting():
    #         fused, foreach = _default_to_fused_or_foreach(
    #             params, differentiable=False, use_fused=False
    #         )
    #     else:
    #         foreach = False
    #         fused = False
    # if foreach is None:
    #     foreach = False
    # if fused is None:
    #     fused = False

    # if foreach and torch.jit.is_scripting():
    #     raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    # if fused and torch.jit.is_scripting():
    #     raise RuntimeError("torch.jit.script not supported with fused optimizers")

    # if foreach and not torch.jit.is_scripting():
    #     func = _multi_tensor_sgd
    # elif fused and not torch.jit.is_scripting():
    #     func = _fused_sgd
    # else:
    #     func = _single_tensor_sgd
    if not foreach and not fused:
        func = _single_tensor_sgd_leak
    else:
        raise NameError("foreach and fused not implemented.")

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        sigma=sigma,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )



def _single_tensor_sgd_leap(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    sigma: float,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf
        lrs = torch.randn_like(grad).mul(lr * sigma).add(lr)
        param.add_(grad.mul(lrs), alpha=-1)
