import math

import torch
from torch import nn, Tensor

import logging

class MNISTReLU(nn.Module):
    def __init__(self, input_dim, input_width, hidden_width, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, input_width)
        self.hidden_fc = nn.Linear(input_width, hidden_width)
        self.output_fc = nn.Linear(hidden_width, output_dim)
        self.bias_on = True
        self.slope_positive = None
        self.slope_negative = None

    def forward_(self, x):
        return self.forward(x)

    def forward(self, x):
        #x = [batch size, height * width]
        h_1 = self.PReLU(self.input_fc(x))
        #h_1 = [batch size, INPUT_WIDTH]
        h_2 = self.PReLU(self.hidden_fc(h_1))
        #h_2 = [batch size, HIDDEN_WIDTH]
        y_pred = self.output_fc(h_2)
        #y_pred = [batch size, output dim]
        return y_pred

    def set_slopes(self, slope_positive = 1.0, slope_negative = 0.25):
        self.slope_positive = slope_positive
        self.slope_negative = slope_negative

    def PReLU(self, input: Tensor) -> Tensor:
        input = torch.where(input >= 0, self.slope_positive * input, self.slope_negative * input)
        return input

    def PReLUz(self, input: float) -> float:
        return self.slope_positive * input if input >= 0 else self.slope_negative * input

    def init_weights(self, cb=0.0, cw=1.0):
        logging.debug("FeedForwardNet weights initialisation with cb={}, cw={}".format(cb, cw))

        #Weight initialisation as in 2.19, 2.20
        self.cb, self.cw = cb, cw
        self.init_linear_weights(self.input_fc, self.bias_on, cb, cw/self.input_fc.in_features)
        self.init_linear_weights(self.hidden_fc, self.bias_on, cb, cw/self.hidden_fc.in_features)
        self.init_linear_weights(self.output_fc, self.bias_on, cb, cw/self.output_fc.in_features)

    def activation_derivative(self, xx):
        '''calculate the derivative of relu'''
        return torch.where(xx>0, self.slope_positive, self.slope_negative)

    @staticmethod
    def init_linear_weights(linear, bias_on, var_b=0.0, var_w=1.0):
        nn.init.normal_(linear.weight, mean = 0., std = math.sqrt(var_w))
        if bias_on:
            nn.init.normal_(linear.bias, mean = 0., std = math.sqrt(var_b))
