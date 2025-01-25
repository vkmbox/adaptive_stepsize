class MetaData:
    def __init__(self, batch_size = 96, output_dim = 10, reduction='mean', device='cpu', lb = 1e-2, lw = 7.5):
        self.batch_size = batch_size
        self.lb = lb
        self.lw = lw
        self.reduction=reduction
        self.device = device
        self.output_dim = output_dim

    #lambdas_w as from (8.5)
    '''
    def lw_input(self):
        return self.lw/self.input_dim
    
    def lw_hidden(self):
        return self.lw/self.input_width

    def lw_output(self):
        return self.lw/self.hidden_width
    '''