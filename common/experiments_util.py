from torch import optim

from optim.optimiser_netline_v3 import NetLine
from optim.optimiser_dz_v3 import NetDz
from optim.scheduler_netline import CosineAnnealingNetLine
from optim.optimizer_pack1 import Lookahead, LEAP
from experiment import ExperimentScheduler as ExperimentWithScheduler

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
    sgd_opt = optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
    leap_opt = LEAP(sgd_opt, sigma=sigma_coeff)
    sgd_sch = optim.lr_scheduler.CosineAnnealingLR(leap_opt, T_max=epochs_per_experiment)
    return leap_opt, sgd_sch

def initialise_sgd(model, epochs_per_experiment, lr_max, momentum, weight_decay):
    print(f"Sgd is initialising with params: lr_max={lr_max}, momentum={momentum}, weight_decay={weight_decay}")
    sgd_opt = optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
    sgd_sch = optim.lr_scheduler.CosineAnnealingLR(sgd_opt, T_max=epochs_per_experiment)
    return sgd_opt, sgd_sch

def initialise_netline_experiment(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay, linearity_coeff,\
                       la_steps = 5, la_alpha = 1.0):
    nl_opt, nl_sch = initialise_netline(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay,\
                                        linearity_coeff=linearity_coeff, la_alpha=la_alpha, la_steps=la_steps)
    return ExperimentWithScheduler("netline", model, nl_opt, nl_sch, do_optimiser_step=False)

def initialise_dz_experiment(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay, linearity_coeff,\
                  la_steps = 5, la_alpha = 1.0):
    dz_opt, dz_sch = initialise_dz(model, meta, epochs_per_experiment, lr1, lr_max, momentum, weight_decay, \
                                   linearity_coeff=linearity_coeff, la_alpha=la_alpha, la_steps=la_steps)
    return ExperimentWithScheduler("dz", model, dz_opt, dz_sch, do_optimiser_step=False)

def initialise_lookahead_experiment(model, epochs_per_experiment, lr_max, momentum, weight_decay, la_steps=5, la_alpha=0.8):
    la_opt, la_sch = initialise_lookahead(model, epochs_per_experiment, lr_max, momentum, weight_decay, la_alpha=la_alpha, la_steps=la_steps)
    return ExperimentWithScheduler("lookahead", model, la_opt, la_sch, do_optimiser_step=True)

def initialise_leap_experiment(model, epochs_per_experiment, lr_max, momentum, weight_decay, sigma_coeff):
    leap_opt, leap_sch = initialise_leap(model, epochs_per_experiment, lr_max, momentum, weight_decay, sigma_coeff)
    return ExperimentWithScheduler("leap", model, leap_opt, leap_sch, do_optimiser_step=True)

def initialise_sgd_experiment(model, epochs_per_experiment, lr_max, momentum, weight_decay):
    sgd_opt, sgd_sch = initialise_sgd(model, epochs_per_experiment, lr_max, momentum, weight_decay)
    return ExperimentWithScheduler("sgd-cosine", model, sgd_opt, sgd_sch, do_optimiser_step=True)
