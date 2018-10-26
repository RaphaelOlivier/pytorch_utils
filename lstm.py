from dropout import LockedDropout


class AdvancedLSTM(nn.LSTM):
    """
    Wrapper on the LSTM class, with learned initial state
    """

    def __init__(self, *args, **kwargs):
        super(AdvancedLSTM, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.h0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(-1, n, -1).contiguous(),
            self.c0.expand(-1, n, -1).contiguous()
        )

    def forward(self, input, hx=None):
        if hx is None:
            n = input.batch_sizes[0]
            hx = self.initial_state(n)
        return super(AdvancedLSTM, self).forward(input, hx=hx)


class PyramidalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, dropout_rate=0, stride=2, method="avg", stride_last=False):
        super(PyramidalLSTM, self).__init__()
        lstms = []
        self.method = method
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.input_size_intern = self.output_size * (stride if self.method == "cat" else 1)
        lstms.append(AdvancedLSTM(input_size, hidden_size, bidirectional=bidirectional))
        for i in range(num_layers-1):
            lstms.append(AdvancedLSTM(self.input_size_intern,
                                      hidden_size, bidirectional=bidirectional))
        self.rnns = nn.ModuleList(lstms)
        self.dropout = nn.Dropout(dropout_rate)
        self.stride = stride
        self.num_layers = num_layers

        self.max_stride_steps = (num_layers-1) if not stride_last else num_layers
        self.modulo = np.power(stride, self.max_stride_steps)
        if self.method == "avg":
            self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride, padding=0)

    def strider(self, x):
        batch_size = x.size(1)
        if self.method == "cat":
            x = x.transpose(0, 1).contiguous().view(
                batch_size, -1, self.input_size_intern).transpose(0, 1)
        else:
            x = self.pool(x.transpose(
                0, 1).transpose(1, 2)).transpose(1, 2).transpose(0, 1)
        return x

    def forward(self, sequences):
        """
        Assumes sequences is a list of sequences sorted by decreasng length
        """
        batch_size = len(sequences)
        lens = [len(s) for s in sequences]
        lens = [n - (n % self.modulo) for n in lens]
        inputs = [sequences[i][:lens[i]] for i in range(batch_size)]
        inputs = rnn.pad_sequence(inputs)
        # print(inputs.size())
        for i in range(self.num_layers):
            packed_input = rnn.pack_padded_sequence(inputs, lens)
            packed_output, _ = self.rnns[i](packed_input)
            padded_output, _ = rnn.pad_packed_sequence(packed_output)
            padded_output = self.dropout(padded_output)
            if i < self.max_stride_steps:
                padded_output = self.strider(padded_output)
                lens = [n // self.stride for n in lens]
                inputs = padded_output
                # print(inputs.size())

        return padded_output.contiguous(), lens


class AdvancedLSTMCell(nn.LSTMCell):
    # Extend LSTMCell to learn initial state
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTMCell, self).__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(n, -1).contiguous(),
            self.c0.expand(n, -1).contiguous()
        )


class MultipleLSTMCells(nn.Module):

    def __init__(self, nlayers, lstm_sizes, dropout_vertical=0, dropout_horizontal=0, residual=False):
        super(MultipleLSTMCells, self).__init__()
        self.nlayers = nlayers
        self.lstm_sizes = lstm_sizes
        self.cells = nn.ModuleList([AdvancedLSTMCell(lstm_sizes[i], lstm_sizes[i + 1])
                                    for i in range(nlayers)])
        self.dropouts = nn.ModuleList(
            [LockedDropout(dropout_horizontal, lstm_sizes[i + 1]) for i in range(nlayers)])
        self.dr = nn.Dropout(dropout_vertical)
        self.residual = residual
        if self.residual:
            self.where_residual = []
            one_residual = False
            for i in range(nlayers):
                if self.lstm_sizes[i] == self.lstm_sizes[i+1]:
                    self.where_residual.append(True)
                    one_residual = True
                else:
                    self.where_residual.append(False)
            if not one_residual:
                raise("Need similar layer sizes to have residual cells")

    def sample_masks(self):

        for i in range(self.nlayers):
            self.dropouts[i].sample_mask()

    def forward(self, input, previous_state=None):
        if previous_state is None:
            previous_state = [c.initial_state() for c in self.cells]

        new_state = previous_state[0].new_full(previous_state[0].size(), 0), \
            previous_state[1].new_full(previous_state[1].size(), 0)

        h = input
        for i in range(self.nlayers):

            residual = h
            h, c = self.cells[i](h, (previous_state[0][i], previous_state[1][i]))
            if self.residual and self.where_residual[i]:
                h = residual + h
            new_state[0][i] = self.dropouts[i](h)
            new_state[0][i] = c
            h = self.dr(h)

        return h, new_state
