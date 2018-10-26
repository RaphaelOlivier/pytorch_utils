
class LockedDropout(nn.Module):

    def __init__(self, dropout_rate, hidden_size):
        super(LockedDropout, self).__init__()

        self.p = dropout_rate
        self.hidden_size = hidden_size
        self.mask = torch.autograd.Variable(torch.Tensor(
            np.ones((1, hidden_size))), requires_grad=False).float().cuda()

    def sample_mask(self):
        new_mask = torch.Tensor((np.random.rand(1, self.hidden_size) < self.p) /
                                self.p) if self.p > 0 else torch.ones(1, self.hidden_size)
        self.mask = torch.nn.Parameter(new_mask, requires_grad=False).float().cuda()

    def forward(self, h):
        if self.training:
            return h * self.mask.expand(h.size())
        else:
            return h
