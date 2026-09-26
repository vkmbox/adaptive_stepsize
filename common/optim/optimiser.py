import torch
from torch import Tensor, optim

import abc
from typing import Any, TypeAlias, List, Optional
from collections import defaultdict
from collections.abc import Iterable

from common.optim.util import AverageCyclicQueue

ParamsT: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, torch.Tensor]]
)

class NetOptimizer(optim.Optimizer, metaclass=abc.ABCMeta):
    def __init__(self, params: ParamsT, defaults: dict[str, Any], lookahead_steps: int, lookahead_alpha: float) -> None:
        super().__init__(params, defaults)

        self.la_alpha = lookahead_alpha
        self._total_la_steps = lookahead_steps
        self.la_states = defaultdict(list) #: defaultdict[int, List[Tensor]] = []
        self.la_backups = defaultdict(list) #: defaultdict[int, List[Tensor]] = []
        self._la_step = 0  # counter for inner optimizer

        self.lr_averaging_queue_size = 50
        self._lr_averaging_queue = None

        self.init_lookahead()

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

    @torch.no_grad()
    def batch_prestep(self):

        for group in self.param_groups:
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
        res = self._batch_step(images, labels, logitsG, eta_target = eta_target, fixed_step = fixed_step)

        do_lookahead = False
        if self.la_alpha < 1.0:
            self._la_step += 1
            if self._la_step >= self._total_la_steps:
                self._la_step, do_lookahead = 0, True

        if do_lookahead:
            for gr_num, group in enumerate(self.param_groups):
                params: List[Tensor] = []
                grads: List[Tensor] = []
                momentum_buffer_list: List[Optional[Tensor]] = []

                la_state = self.la_states[gr_num]
                has_sparse_grad = self._init_group(
                    group, params, grads, momentum_buffer_list
                )
                if group["foreach"] == False:
                    for num, param in enumerate(params):
                        # Lookahead and cache the current optimizer parameters
                        param_state = la_state[num]
                        if self.la_alpha != 1.0:
                            param.mul_(self.la_alpha).add_(param_state, alpha=1.0 - self.la_alpha)
                        param_state.copy_(param)
                else:
                    if self.la_alpha != 1.0:
                        torch._foreach_mul_(params, self.la_alpha)
                        torch._foreach_add_(params, la_state, alpha=1.0 - self.la_alpha)
                    torch._foreach_copy_(la_state, params)

        return res

    @abc.abstractmethod
    def _batch_step(self, images, labels, logitsG, eta_target, fixed_step):
        '''Methid is expected to call from batch_step only
        '''
        pass

    def _step_results(self, eta, eta2_pre, alpha, alpha_nomomentum, qq1):
        result = {}
        result['eta'] = eta
        result['eta2_pre'] = eta2_pre
        result['alpha'] = alpha
        result['alpha_nomomentum'] = alpha_nomomentum
        result['qq1'] = qq1
        return result

    def init_lookahead(self):
        # Cache the current optimizer parameters params.append(p)
        for gr_num, group in enumerate(self.param_groups):
            la_state = self.la_states[gr_num]
            for param in group['params']:
                param_state = torch.zeros_like(param)
                param_state.copy_(param)
                la_state.append(param_state)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        self.la_backups.clear()
        for gr_num, group in enumerate(self.param_groups):
            la_backup = self.la_backups[gr_num]
            la_state = self.la_states[gr_num]
            for num, param in enumerate(group['params']):
                backup_params = torch.zeros_like(param.data)
                backup_params.copy_(param.data)
                la_backup.append(backup_params)
                param.data.copy_(la_state[num])

    def _clear_and_load_backup(self):
        for gr_num, group in enumerate(self.param_groups):
            la_backup = self.la_backups[gr_num]
            for num, param in enumerate(group['params']):
                param.data.copy_(la_backup[num])
        self.la_backups.clear()

    def _step1(self):
        grads_final = defaultdict(dict)

        #Small step
        for gr_num, group in enumerate(self.param_groups):
            params: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_buffer_list: list[Tensor | None] = []

            maximize=group["maximize"]
            foreach=group["foreach"]
            weight_decay=group["weight_decay"]
            eta1=group["lr1"]
            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            if foreach:
                if maximize:
                    grads = torch._foreach_neg(grads)
                if weight_decay != 0:
                    grads = torch._foreach_add(grads, params, alpha=weight_decay)

                #buf = []
                #for grad in grads: buf.append(grad.detach().clone())
                grads_final[gr_num] = grads
                params_shift1 = torch._foreach_mul(grads_final[gr_num], eta1)
                torch._foreach_sub_(params, params_shift1)

            else:
                grad_final = grads_final[gr_num]
                for num, param in enumerate(params):
                    grad = grads[num] if not maximize else -grads[num]

                    if weight_decay != 0:
                        grad = grad.add(param, alpha=weight_decay)

                    grad_final[num] = grad #.detach().clone()
                    param_shift1 = grad_final[num].mul(eta1)
                    param.sub_(param_shift1)

        return grads_final

    def _step2(self, grads_final, eta1, eta2):
        eta2_shift = eta2.add(-eta1)
        for gr_num, group in enumerate(self.param_groups):
            group = self.param_groups[0]
            params: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_buffer_list: list[Tensor | None] = []

            foreach=group["foreach"]
            momentum = group["momentum"]
            grad_final = grads_final[gr_num]
            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            if foreach:
                params_shift2 = torch._foreach_mul(grads_final[gr_num], eta2_shift)
                torch._foreach_sub_(params, params_shift2)
            else:
                for num, param in enumerate(params):
                    param_shift2 = grad_final[num].mul(eta2_shift)
                    param.sub_(param_shift2)

            if momentum != 0:
                for num, param in enumerate(params):
                    buf = momentum_buffer_list[num]
                    momentum_shift = grad_final[num].mul(eta2)
                    if buf is None:
                        buf = momentum_shift.detach().clone()
                    else:
                        buf.add_(momentum_shift)
                    # update momentum_buffers in state
                    stat = self.state[param]
                    stat["momentum_buffer"] = buf
