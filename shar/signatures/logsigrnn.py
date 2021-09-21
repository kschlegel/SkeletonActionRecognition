import numpy as np
import torch
import torch.nn as nn
import signatory


class LogSigRNN(nn.Module):
    """
    Implement the Logsig-RNN module.

    Takes an input sequence of shape
        (batch, frames, in_channels).
    The first step is to split the sequence at the temporal dimension
    into n segments. At each segment, computes the logsignature up to
    level k. To use the information of the start points of each segment,
    the start point is concatenated to the output of the log-signature module.
    Then the frequency reduced sequence is passed into a LSTM.
    Returns a tensor of shape
        (batch, n_segments, n_hiddens)
    """
    def __init__(self,
                 in_channels: int,
                 logsig_depth: int,
                 n_segments: int,
                 n_hiddens: int,
                 include_startpoint: bool = True) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        logsig_depth : int
            The level of the log-signature to compute
        n_segments : int
            The number of segments to split the input sequence into
        n_hiddens : int
            The number of hidden neurons in the LSTM
        include_startpoint : bool, optional (default is true)
            If True each segments logsignature is concatenated with the start
            point of the segment to include the positional information in later
            stages
        """
        super(LogSigRNN, self).__init__()

        logsig_channels = signatory.logsignature_channels(
            in_channels=in_channels, depth=logsig_depth)
        self.logsig = _LogSig(in_channels=in_channels,
                              logsig_depth=logsig_depth,
                              n_segments=n_segments,
                              include_startpoint=include_startpoint)

        self.lstm = nn.LSTM(input_size=in_channels + logsig_channels,
                            hidden_size=n_hiddens,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, frames, in_channels)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, n_segments, n_hiddens)
        """
        N, T, C = x.size()

        x_logsig = self.logsig(x).type_as(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x_logsig)

        return x


class _LogSig(nn.Module):
    """
    Compute the segment-wise log-signatures.

    Takes an input sequence of shape
        (batch, frames, channels),
    splits it into n segments and computes the logsignature
    of each segment. Returns an output sequence of shape
        (batch, n_segments, logsig_channels)
    """
    def __init__(self,
                 logsig_depth: int,
                 n_segments: int,
                 include_startpoint: bool = True) -> None:
        """
        Parameters
        ----------
        logsig_depth : int
            The level of the log-signature to compute
        n_segments : int
            The number of segments to split the input sequence into
        include_startpoint : bool, optional (default is true)
            If True each segments logsignature is concatenated with the start
            point of the segment to include the positional information in later
            stages
        """
        super(_LogSig, self).__init__()
        self.n_segments = n_segments

        self.logsignature = signatory.LogSignature(depth=logsig_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, stream, in_channels)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, n_segments, logsig_channels)
        """
        nT = x.size(1)
        t_vec = np.linspace(1, nT, self.n_segments + 1)
        t_vec = [int(round(x) - 1) for x in t_vec]

        MultiLevelLogSig = []
        for i in range(self.n_segments):
            MultiLevelLogSig.append(
                self.logsignature(x[:, t_vec[i]:t_vec[i + 1] +
                                    1, :].clone()).unsqueeze(1))
        x_logsig = torch.cat(MultiLevelLogSig, axis=1)

        out = torch.cat([x_logsig, x[:, t_vec[:-1]].clone()], axis=-1)

        return out
