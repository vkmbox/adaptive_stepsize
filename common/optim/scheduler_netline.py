from torch import optim
from common.optim.util import cosine_annealing2_lr, line_annealing2_lr

class CosineAnnealingNetLine:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        epochs_per_experiment: int,
        epochs_warmup: int,
        epochs_shutdown: int,
        lr_max: float = 2e-2
    ) -> None:
        if lr_max < 0.0:
            raise ValueError(f"Invalid learning rate: {lr_max}")
        
        self.optimizer = optimizer
        self.epochs_per_experiment = epochs_per_experiment
        self.epochs_warmup = epochs_warmup
        self.epochs_shutdown = epochs_shutdown
        self.lr_max = lr_max

        self._epoch = 0
        self.eta_target = lr_max
        self._epoch_fixed_step = False

        self.step(-1)

    def step(self, epoch = None):
        """Method to call in every epoch for scheduler/optimiser params adjustment. It is called after the epoch's training loop
        """
        if epoch is not None:
            self._epoch = epoch

        self._epoch += 1

        EPOCHS_PER_EXPERIMENT = self.epochs_per_experiment
        self.eta_target = cosine_annealing2_lr(self.lr_max, 0.0, 0, EPOCHS_PER_EXPERIMENT, self._epoch)

        #self._arctan_coeff = line_annealing2_lr(4.0, 20.0, EPOCHS_PER_EXPERIMENT/3, EPOCHS_PER_EXPERIMENT*2/3, self._epoch)
        self._arctan_coeff = line_annealing2_lr(20.0, 10.0, EPOCHS_PER_EXPERIMENT/3, EPOCHS_PER_EXPERIMENT*2/3, self._epoch)

        self._epoch_fixed_step = self._epoch < self.epochs_warmup or self._epoch >= (self.epochs_per_experiment - self.epochs_shutdown)

    def batch_prestep(self):
        self.optimizer.batch_prestep()

    def batch_step(self, x, y, y_pred, **kwargs):
        return self.optimizer.batch_step\
            (x, y, y_pred, eta_target = self.eta_target, epoch_fixed_step = self._epoch_fixed_step, arctan_coeff = self._arctan_coeff)
