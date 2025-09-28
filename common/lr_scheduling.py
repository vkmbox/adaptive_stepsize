import math

def cosine_annealing_lr(eta0, eta1, epoch_cos_start, epoch_finish, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    return eta1 + 0.5*(eta0-eta1)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(epoch_finish-epoch_cos_start)))

def cosine_annealing2_lr(eta0, eta1, epoch_cos_start, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta1
    return eta1 + 0.5*(eta0-eta1)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(epoch_cos_finish-epoch_cos_start)))

def cosine_annealing2half_lr(eta0, eta1, epoch_cos_start, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta1
    return eta1 + 0.5*(eta0-eta1)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(2*(epoch_cos_finish-epoch_cos_start))))

def cosine_annealing3_lr(eta0, eta1, eta2, epoch_cos_start, epoch_cos_middle, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta2
    if epoch_cos_start <= epoch_curr and epoch_curr < epoch_cos_middle:
        return eta1 + 0.5*(eta0-eta1)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(epoch_cos_middle-epoch_cos_start)))
    return eta2 + 0.5*(eta1-eta2)*(1+math.cos((epoch_curr-epoch_cos_middle)*math.pi/(epoch_cos_finish-epoch_cos_middle)))

def exp_cosine_annealing3_lr(eta0, eta1, eta2, exp_range, epoch_exp_start, epoch_cos_middle, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_exp_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta2
    if epoch_exp_start <= epoch_curr and epoch_curr < epoch_cos_middle:
        return eta1 + (eta0-eta1)*math.exp(-exp_range*(epoch_curr-epoch_exp_start)/(epoch_cos_middle-epoch_exp_start))
    return eta2 + 0.5*(eta1-eta2)*(1+math.cos((epoch_curr-epoch_cos_middle)*math.pi/(epoch_cos_finish-epoch_cos_middle)))

def exp_cosine_annealing4_lr(eta0, eta1, eta2, eta3, exp_range, epoch_exp_start, epoch_cos_start, epoch_cos_middle, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_exp_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta3
    if epoch_exp_start <= epoch_curr and epoch_curr < epoch_cos_start:
        return eta1 + (eta0-eta1)*math.exp(-exp_range*(epoch_curr-epoch_exp_start)/(epoch_cos_start-epoch_exp_start))
    if epoch_cos_start <= epoch_curr and epoch_curr < epoch_cos_middle:
        return eta2 + 0.5*(eta1-eta2)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(epoch_cos_middle-epoch_cos_start)))
    return eta3 + 0.5*(eta2-eta3)*(1+math.cos((epoch_curr-epoch_cos_middle)*math.pi/(epoch_cos_finish-epoch_cos_middle)))

def exp_cosine_annealing4_lr2(eta0, eta1, eta2, eta3, exp_range, epoch_exp_start, epoch_cos_middle1, epoch_cos_middle2, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_exp_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta3
    if epoch_exp_start <= epoch_curr and epoch_curr < epoch_cos_middle1:
        return eta1 + (eta0-eta1)*math.exp(-exp_range*(epoch_curr-epoch_exp_start)/(epoch_cos_middle1-epoch_exp_start))
    if epoch_cos_middle1 <= epoch_curr and epoch_curr < epoch_cos_middle2:
        return eta2 + 0.5*(eta1-eta2)*(1+math.cos((epoch_curr-epoch_cos_middle1)*math.pi/(epoch_cos_middle2-epoch_cos_middle1)))
    return eta3 + (2/(2+math.sqrt(2)))*(eta2-eta3)*((1/math.sqrt(2))+math.cos((epoch_curr-epoch_cos_middle2)*math.pi*0.75/(epoch_cos_finish-epoch_cos_middle2)))

def cosine_annealing2_lr2(eta0, eta1, epoch_cos_start, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta1
    return eta1 + (2/(2+math.sqrt(2)))*(eta0-eta1)*((1/math.sqrt(2))+math.cos((epoch_curr-epoch_cos_start)*math.pi*0.75/(epoch_cos_finish-epoch_cos_start)))

def cosine_annealing4_lr(eta0, eta1, eta2, eta3, epoch_cos_pre, epoch_cos_start, epoch_cos_middle, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_pre:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta3
    if epoch_cos_pre <= epoch_curr and epoch_curr < epoch_cos_start:
        return eta1 + 0.5*(eta0-eta1)*(1+math.cos((epoch_curr-epoch_cos_pre)*math.pi/(epoch_cos_start-epoch_cos_pre)))
    if epoch_cos_start <= epoch_curr and epoch_curr < epoch_cos_middle:
        return eta2 + 0.5*(eta1-eta2)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(epoch_cos_middle-epoch_cos_start)))
    return eta3 + 0.5*(eta2-eta3)*(1+math.cos((epoch_curr-epoch_cos_middle)*math.pi/(epoch_cos_finish-epoch_cos_middle)))
