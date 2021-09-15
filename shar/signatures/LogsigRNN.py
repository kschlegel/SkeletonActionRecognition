import numpy as np
import torch
import torch.nn as nn
import signatory


class LogSig_RNN(nn.Module):
    """
    Implement the Logsig-RNN algorithm 

    The input sequence of shape (batch, frames, in_channels).
    The first step is to split the sequence at the temporal dimension
    into n segments. At each segment, computes the logsignature up to 
    level k. To use the information of start points of each segment,
    the start point is concatenated to the output of the logsignature.
    Then the frequency reduced sequence is put into a LSTM.
    Returns a tensor of shape
        (batch, n_segments, n_hiddens)
    """
    def __init__(self, in_channels:int, logsig_depth:int, n_segments:int, n_hiddens:int):
        """
        in_channels: int
            input channels
        logsig_depth: int
            logsignature level k
        n_segments: int
            number of segments
        n_hiddens: int
            number of hidden neurons in LSTM
        """
        super(LogSig_RNN, self).__init__()
        self.n_segments = n_segments

        self.logsig_channels = signatory.logsignature_channels(
            in_channels=in_channels, depth=logsig_depth)
        self.logsig = LogSig_(in_channels, logsig_depth,
                              logsig_channels=self.logsig_channels)
        self.start_position = start_position_()

        self.lstm = nn.LSTM(
            input_size=in_channels + self.logsig_channels,
            hidden_size=n_hiddens,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        # inp is a three dimensional tensor of shape (batch, frames, in_channels)
        N, T, C = x.size()

        x_sp = self.start_position(x, self.n_segments).type_as(x)
        x_logsig = self.logsig(x, self.n_segments).type_as(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(torch.cat([x_logsig, x_sp], axis=-1))

        return x


class LogSig_(nn.Module):
    """
    Splits the input into n segments and computes the logsignature
    of each segment.
    The input sequence of shape (batch, frames, channels).
    Returns output of shape (batch, n_segments, logsig_channels)
    """
    def __init__(self, in_channels, logsig_depth, logsig_channels):
        """
        in_channels: int
            input channels
        logsig_depth: int
            logsignature level k
        logsig_channels: int
            logsignature channels
        """
        super(LogSig_, self).__init__()
        self.in_channels = in_channels
        self.logsig_depth = logsig_depth

        self.logsignature = signatory.LogSignature(depth=logsig_depth)

        self.logsig_channels = logsig_channels

    def forward(self, inp, n_segments):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        nT = inp.size(1)
        dim_path = inp.size(-1)
        t_vec = np.linspace(1, nT, n_segments + 1)
        t_vec = [int(round(x)) for x in t_vec]

        MultiLevelLogSig = []
        for i in range(n_segments):
            MultiLevelLogSig.append(self.logsignature(
                inp[:, t_vec[i] - 1:t_vec[i + 1], :].clone()).unsqueeze(1))
        out = torch.cat(MultiLevelLogSig, axis=1)
        return out


class start_position_(nn.Module):
    """
    Extract the starting points of each segment
    """
    def __init__(self, ):
        super(start_position_, self).__init__()

    def forward(self, inp, n_segments):
        nT = inp.size(1)
        t_vec = np.linspace(1, nT, n_segments + 1)
        t_vec = [int(round(x)) - 1 for x in t_vec]
        return inp[:, t_vec[:-1]].clone()
