import warnings

import torch

from ._signaturelayers import _SegmentSignatures, _SegmentSignaturesSequential


class SegmentSignatures(torch.nn.Module):
    """
    Take signatures of temporal segments of each input stream.

    Takes an input sequence of shape
        (batch, in_channels, frames, nodes).
    Split the sequence at the temporal dimension into n segments. For each
    segment of each node, computes the signature or logsignature up to
    level k.
    The number of signature channels in the output can be accessed via
        model.out_channels
    Returns a tensor of shape
        (batch, signatue_channels, num_segments, nodes)
    """
    def __init__(self,
                 in_channels: int,
                 signature_lvl: int,
                 num_segments: int,
                 logsignature: bool = False,
                 parallelize_signatures: bool = True) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        signature_lvl : int
            The level of the log-signature to compute
        num_segments : int
            The number of segments to split the input sequence into
        logsignature : bool, optional (default is False)
            If set to True the logsignature of each segment is computed instead
            of the signature.
        parallelize_signatures : bool, optional (default is True)
            If True the segments are moved into the batch dimension to compute
            signatures of all segments in parallel. If False then segments are
            processed sequentially. Computing signatures in parallel requires
            at least PyTorch version 1.7.0, for older versions of PyTorch
            defaults to sequential computation.
        """
        super().__init__()
        self._num_segments = num_segments

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

        self.signatures = SignatureModel(in_channels=in_channels,
                                         signature_lvl=signature_lvl,
                                         num_segments=num_segments,
                                         logsignature=logsignature,
                                         include_startpoint=False)
        self.out_channels = self.signatures.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, nodes)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch,self.out_channels,num_segments,nodes)
        """
        batch, in_channels, frames, nodes = x.size()
        x = x.permute(0, 3, 1, 2)
        x = torch.reshape(x, (batch * nodes, in_channels, frames))

        sigs = self.signatures(x)
        # x = (batch*nodes, sig_channels, num_segments)

        sigs = torch.reshape(sigs, (batch, nodes, -1, self._num_segments))
        # (batch, nodes, sig_channels, num_segments)
        sigs = sigs.permute(0, 2, 3, 1)

        return sigs
