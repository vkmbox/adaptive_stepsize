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

def cosine_range_annealing2_lr(eta0, eta1, epoch_cos_start, epoch_cos_finish, radian_start, radian_end, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta1
    
    radian_value = radian_start + (epoch_curr-epoch_cos_start)*(radian_end - radian_start)/(epoch_cos_finish-epoch_cos_start)
    cos_scale = 1/(math.cos(radian_start) - math.cos(radian_end))
    return eta1 + cos_scale*(eta0-eta1)*(math.cos(radian_value)-math.cos(radian_end))

def line_annealing2_lr(eta0, eta1, epoch_line_start, epoch_line_finish, epoch_curr):
    if epoch_curr < epoch_line_start:
        return eta0
    if epoch_line_finish <= epoch_curr:
        return eta1
    return eta0 + (eta1-eta0)*(epoch_curr-epoch_line_start)/(epoch_line_finish-epoch_line_start)

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

def line_annealing3_lr(eta0, eta1, eta2, epoch_cos_start, epoch_cos_middle, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_start:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta2
    if epoch_cos_start <= epoch_curr and epoch_curr < epoch_cos_middle:
        return eta0 + (eta1-eta0)*((epoch_curr-epoch_cos_start)/(epoch_cos_middle-epoch_cos_start))
    #return eta2 + 0.5*math.sqrt(2)*(eta1-eta2)*((1/math.sqrt(2))+math.cos((math.pi/4)+((epoch_curr-epoch_cos_middle)*math.pi/(2*(epoch_cos_finish-epoch_cos_middle)))))
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

def line_annealing4_lr(eta0, eta1, eta2, eta3, epoch_pre, epoch_start, epoch_middle, epoch_finish, epoch_curr):
    if epoch_curr < epoch_pre:
        return eta0
    if epoch_finish <= epoch_curr:
        return eta3
    if epoch_pre <= epoch_curr and epoch_curr < epoch_start:
        return eta0 + (eta1-eta0)*((epoch_curr-epoch_pre)/(epoch_start-epoch_pre))
    if epoch_start <= epoch_curr and epoch_curr < epoch_middle:
        return eta1 + (eta2-eta1)*((epoch_curr-epoch_start)/(epoch_middle-epoch_start))
    return eta2 + (eta3-eta2)*((epoch_curr-epoch_middle)/(epoch_finish-epoch_middle))

def line_cosine_annealing4_lr(eta0, eta1, eta2, eta3, epoch_cos_pre, epoch_cos_start, epoch_cos_middle, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_pre:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta3
    if epoch_cos_pre <= epoch_curr and epoch_curr < epoch_cos_start:
        return eta0 + (eta1-eta0)*((epoch_curr-epoch_cos_pre)/(epoch_cos_start-epoch_cos_pre))
    if epoch_cos_start <= epoch_curr and epoch_curr < epoch_cos_middle:
        return eta1 + (eta2-eta1)*((epoch_curr-epoch_cos_start)/(epoch_cos_middle-epoch_cos_start))
    return eta3 + 0.5*(eta2-eta3)*(1+math.cos((epoch_curr-epoch_cos_middle)*math.pi/(epoch_cos_finish-epoch_cos_middle)))

'''
def line_annealing4_lr(eta0, eta1, eta2, eta3, epoch_cos_pre, epoch_cos_start, epoch_cos_middle, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_pre:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta3
    if epoch_cos_pre <= epoch_curr and epoch_curr < epoch_cos_start:
        return eta0 + (eta1-eta0)*((epoch_curr-epoch_cos_pre)/(epoch_cos_start-epoch_cos_pre))
    if epoch_cos_start <= epoch_curr and epoch_curr < epoch_cos_middle:
        return eta1 + (eta2-eta1)*((epoch_curr-epoch_cos_start)/(epoch_cos_middle-epoch_cos_start))
    return eta2 + (eta3-eta2)*((epoch_curr-epoch_cos_middle)/(epoch_cos_finish-epoch_cos_middle))
'''

def cosine_annealing6_lr(eta0, eta1, eta2_1, eta2_2, eta2_3, eta3, epoch_cos_pre, epoch_cos_start\
                         , epoch_cos_middle1, epoch_cos_middle2, epoch_cos_middle3, epoch_cos_finish, epoch_curr):
    if epoch_curr < epoch_cos_pre:
        return eta0
    if epoch_cos_finish <= epoch_curr:
        return eta3
    if epoch_cos_pre <= epoch_curr and epoch_curr < epoch_cos_start:
        return eta1 + 0.5*(eta0-eta1)*(1+math.cos((epoch_curr-epoch_cos_pre)*math.pi/(epoch_cos_start-epoch_cos_pre)))
    if epoch_cos_start <= epoch_curr and epoch_curr < epoch_cos_middle1:
        return eta2_1 + 0.5*(eta1-eta2_1)*(1+math.cos((epoch_curr-epoch_cos_start)*math.pi/(epoch_cos_middle1-epoch_cos_start)))
    if epoch_cos_middle1 <= epoch_curr and epoch_curr < epoch_cos_middle2:
        return eta2_2 + 0.5*(eta2_1-eta2_2)*(1+math.cos((epoch_curr-epoch_cos_middle1)*math.pi/(epoch_cos_middle2-epoch_cos_middle1)))
    if epoch_cos_middle2 <= epoch_curr and epoch_curr < epoch_cos_middle3:
        return eta2_3 + 0.5*(eta2_2-eta2_3)*(1+math.cos((epoch_curr-epoch_cos_middle2)*math.pi/(epoch_cos_middle3-epoch_cos_middle2)))
    return eta3 + 0.5*(eta2_3-eta3)*(1+math.cos((epoch_curr-epoch_cos_middle3)*math.pi/(epoch_cos_finish-epoch_cos_middle3)))
