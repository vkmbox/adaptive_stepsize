from torch import optim

from common.optim.optimiser_netline_v3 import NetLine
from common.optim.optimiser_dz_v3 import NetDz
from common.optim.scheduler_netline import CosineAnnealingNetLine
from common.optim.optimizer_pack1 import Lookahead
from common.optim.sgd_leap import SGD_LEAP
from common.cnn import _make_resnet18v2, _make_resnet34v2

def initialise_netline(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay, linearity_coeff,\
                       la_steps = 5, la_alpha = 1.0):
    print(f"Netline is initialising with params: lr1={lr1}, lr_max={lr_max}, momentum={momentum}, "+\
          f"weight_decay={weight_decay}, linearity_coeff={linearity_coeff}, la_steps={la_steps}, la_alpha={la_alpha}")

    nl_opt = NetLine(model=model, meta=meta, lr1=lr1, momentum=momentum, weight_decay=weight_decay, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha, linearity_coeff = linearity_coeff)
    nl_opt.lr_averaging_queue_size = 50
    nl_sch = CosineAnnealingNetLine(optimizer = nl_opt, lr_max=lr_max,\
                                    epochs_per_experiment = epochs_per_experiment, epochs_warmup = 0, epochs_shutdown = 0)
    return nl_opt, nl_sch

def initialise_dz(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay, linearity_coeff,\
                  la_steps = 5, la_alpha = 1.0):
    print(f"Dz is initialising with params: lr1={lr1}, lr_max={lr_max}, momentum={momentum}, "+\
          f"weight_decay={weight_decay}, linearity_coeff={linearity_coeff}, la_steps={la_steps}, la_alpha={la_alpha}")

    dz_opt = NetDz(model=model, meta=meta, lr1=lr1, momentum=momentum, weight_decay=weight_decay, foreach=True,\
                   lookahead_steps=la_steps, lookahead_alpha=la_alpha, linearity_coeff = linearity_coeff)
    dz_opt.lr_averaging_queue_size = 50
    dz_sch = CosineAnnealingNetLine(optimizer = dz_opt, lr_max=lr_max,\
                                    epochs_per_experiment = epochs_per_experiment, epochs_warmup = 0, epochs_shutdown = 0)
    return dz_opt, dz_sch

def initialise_lookahead(model, epochs_per_experiment, lr_max, momentum, weight_decay, la_steps=5, la_alpha=0.8):
    print(f"Lookahead is initialising with params: lr_max={lr_max}, momentum={momentum}, "+\
          f"weight_decay={weight_decay}, la_steps={la_steps}, la_alpha={la_alpha}")

    la_opt_sgd = optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
    la_opt = Lookahead(la_opt_sgd, la_steps=la_steps, la_alpha=la_alpha)
    la_sch = optim.lr_scheduler.CosineAnnealingLR(la_opt, T_max=epochs_per_experiment)
    return la_opt, la_sch

def initialise_leap(model, epochs_per_experiment, lr_max, momentum, weight_decay, sigma_coeff):
    print(f"Leap is initialising with params: lr_max={lr_max}, momentum={momentum}, weight_decay={weight_decay}, "+\
          f"sigma_coeff={sigma_coeff}")
    leap_opt = SGD_LEAP(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay, sigma=sigma_coeff)
    #leap_opt = LEAP(sgd_opt, sigma=sigma_coeff)
    sgd_sch = optim.lr_scheduler.CosineAnnealingLR(leap_opt, T_max=epochs_per_experiment)
    return leap_opt, sgd_sch

def initialise_sgd(model, epochs_per_experiment, lr_max, momentum, weight_decay):
    print(f"Sgd is initialising with params: lr_max={lr_max}, momentum={momentum}, weight_decay={weight_decay}")
    sgd_opt = optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
    sgd_sch = optim.lr_scheduler.CosineAnnealingLR(sgd_opt, T_max=epochs_per_experiment)
    return sgd_opt, sgd_sch

def clone_resnet18v2(model, meta):
    mdl = _make_resnet18v2(meta.output_dim).to(meta.device)
    mdl.load_state_dict(model.state_dict())
    return mdl

def clone_resnet34v2(model, meta):
    mdl = _make_resnet34v2(meta.output_dim).to(meta.device)
    mdl.load_state_dict(model.state_dict())
    return mdl
