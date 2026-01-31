import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

import logging

class CrossEntropyKappaLoss:

    def __init__(
        self,
        output_dim: int,
        kappa_step: float = 0.0,
        do_logging: bool = False
    ) -> None:
        self.output_dim = output_dim
        self.kappa_step = kappa_step
        self.do_logging = do_logging

        self.tensor_zero = torch.tensor(0.0)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            pp = F.one_hot(target, self.output_dim)
            kappa_raw = 1.0 + self.kappa_step*torch.sum(pp*F.log_softmax(input, dim=1), dim=1)
            kappa = torch.maximum(kappa_raw, self.tensor_zero)
            if self.do_logging:
                logging.info("##Kappa: avg={}, min={}, max={}".format(torch.mean(kappa), torch.min(kappa), torch.max(kappa)))

        return torch.mean(kappa*F.cross_entropy(input, target, reduction='none'))

    def forwardWithStats(self, input: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            pp = F.one_hot(target, self.output_dim)
            kappa_raw = 1.0 + self.kappa_step*torch.sum(pp*F.log_softmax(input, dim=1), dim=1)
            kappa = torch.maximum(kappa_raw, self.tensor_zero)
            kappa_avg, kappa_min = torch.mean(kappa), torch.min(kappa)
            if self.do_logging:
                logging.info("##Kappa: avg={}, min={}, max={}".format(kappa_avg, kappa_min, torch.max(kappa)))

        return torch.mean(kappa*F.cross_entropy(input, target, reduction='none')), kappa_avg, kappa_min
