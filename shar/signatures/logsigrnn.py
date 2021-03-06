import warnings

import torch

from ._signaturelayers import _SegmentSignatures, _SegmentSignaturesSequential


class LogSigRNN(torch.nn.Module):
    """
    Implement the Logsig-RNN module.

    Takes an input sequence of shape
        (batch, in_channels, frames, nodes).
    The first step is to split the sequence at the temporal dimension
    into n segments. At each segment, computes the logsignature up to
    level k. To use the information of the start points of each segment,
    the start point is concatenated to the output of the log-signature module.
    Then the frequency reduced sequence is passed into a LSTM with hidden state
    of size out_channels. The output is the hidden state sequence of the LSTM.
    Returns a tensor of shape
        (batch, out_channels, num_segments, nodes)
    """
    def __init__(self,
                 in_channels: int,
                 logsignature_lvl: int,
                 num_segments: int,
                 out_channels: int,
                 include_startpoint: bool = True,
                 parallelize_signatures: bool = True) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        logsignature_lvl : int
            The level of the log-signature to compute
        num_segments : int
            The number of segments to split the input sequence into
        out_channels : int
            The size of the hidden state of the LSTM, which forms the output
            sequence of the LogSigRNN
        include_startpoint : bool, optional (default is true)
            If True each segments logsignature is concatenated with the start
            point of the segment to include the positional information in later
            stages
        parallelize_signatures : bool, optional (default is True)
            If True the segments are moved into the batch dimension to compute
            signatures of all segments in parallel. If False then segments are
            processed sequentially. Computing signatures in parallel requires
            at least PyTorch version 1.7.0, for older versions of PyTorch
            defaults to sequential computation.
        """
        super(LogSigRNN, self).__init__()
        self.num_segments = num_segments

        version = torch.__version__.split(".")
        if int(version[0]) == 1 and int(version[1]) < 7:
            if parallelize_signatures:
                warnings.warn("Computing signatures in parallel requires at "
                              "least PyTorch version 1.7.0. Defaulting to "
                              "sequential signature computation.")
            parallelize_signatures = False
        if parallelize_signatures:
            SignatureModel = _SegmentSignatures
        else:
            SignatureModel = _SegmentSignaturesSequential

        self.logsig = SignatureModel(in_channels=in_channels,
                                     signature_lvl=logsignature_lvl,
                                     num_segments=num_segments,
                                     logsignature=True,
                                     include_startpoint=include_startpoint)

        self.lstm = torch.nn.LSTM(input_size=self.logsig.out_channels,
                                  hidden_size=out_channels,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, nodes)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, out_channels, num_segments, nodes)
        """
        batch, in_channels, frames, nodes = x.size()
        x = x.permute(0, 3, 1, 2)
        x = torch.reshape(x, (batch * nodes, in_channels, frames))

        x_logsig = self.logsig(x).type_as(x)
        x_logsig = x_logsig.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x_logsig)

        x = torch.reshape(x, (batch, nodes, self.num_segments, -1))
        # (batch, nodes, num_segments, channels)
        x = x.permute(0, 3, 2, 1)

        return x
