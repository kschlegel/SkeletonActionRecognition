import math
from typing import Optional

import torch
import signatory


class _SegmentSignatures(torch.nn.Module):
    """
    Compute the segment-wise signatures or log-signatures.

    Takes an input sequence of shape
        (batch, frames, channels),
    splits it along the frame dimension into n segments and computes the
    (log)signature of each segment. Returns an output sequence of shape
        (batch, num_segments, (log)sig_channels)
    Optionally concatenates the start point of each segment to its
    (log)signature, to preserve spatial information (as the signature is
    translation invariant). In this case the output sequence is of shape
        (batch, num_segments, (log)sig_channels+in_channels)
    The class provides the out_channels property containing the number of
    channels in the output sequence (i.e. (log)signature dimension plus
    in_channels if applicable)

    This version moves the segment dimension into the batch dimension for
    parallel signature computation. To achieve this the sequence is padded at
    the end with the last time step before splitting into segments. This means
    the last segment is actually shorter than the set segment length or segment
    length calculated from the number of segments.
    """
    def __init__(self,
                 in_channels: int,
                 signature_lvl: int,
                 num_segments: Optional[int] = None,
                 segment_len: Optional[int] = None,
                 logsignature: bool = False,
                 include_startpoint: bool = False) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels
        signature_lvl : int
            The level of the (log)signature to compute
        num_segments : int, optional (default is None)
            The number of segments to split the input sequence into. If not
            given segment_len must be specified. If given segment_len is
            ignored.
        segment_len : int, optional (default is None)
            The length of each segment the sequence should be split into. If
            not given num_segments must be specified.
        logsignature : bool, optional (default is False)
            If set to True the logsignature of each segment is computed instead
            of the signature.
        include_startpoint : bool, optional (default is False)
            If True each segments logsignature is concatenated with the start
            point of the segment to include the positional information in later
            stages
        """
        super().__init__()
        if num_segments is None and segment_len is None:
            raise ValueError("Either number of segments or segment length "
                             "must be specified.")
        self._num_segments = num_segments
        self._segment_len = segment_len

        # Pick the signature computation fn and set the number of output
        # channels
        if logsignature:
            self._signature = signatory.LogSignature(depth=signature_lvl)
            self.out_channels = signatory.logsignature_channels(
                in_channels=in_channels, depth=signature_lvl)
        else:
            self._signature = signatory.Signature(depth=signature_lvl)
            self.out_channels = signatory.signature_channels(
                channels=in_channels, depth=signature_lvl)

        # include path dimension in output dimension if start points are
        # included
        self._include_startpoint = include_startpoint
        if self._include_startpoint:
            self.out_channels += in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, out_channels, frames)
            out_channels is the dimension of the (log)signature plus the number
            of in_channels if the start point is included.
        """
        batch, channels, frames = x.size()
        x = x.transpose(1, 2)

        # Set number of segments & segment length values depending on settings
        if self._num_segments is not None:
            num_segments = self._num_segments
            segment_len = int(math.ceil(frames / num_segments))
        elif self._segment_len is not None:
            segment_len = self._segment_len
            num_segments = int(math.ceil(frames / segment_len))
        else:
            raise ValueError("Either number of segments or segment length "
                             "must be specified.")

        # Pad sequence at the end with last frame to split into segments
        pad_len = (segment_len - frames % segment_len) % segment_len
        padding = torch.unsqueeze(x[:, -1, :], 1)
        padding = padding.repeat(1, pad_len, 1)
        x = torch.cat((x, padding), dim=1)
        x = torch.reshape(x, (batch * num_segments, segment_len, channels))
        # (b1seg1, b1seg2,...,b2seg1,b2seg2,...)

        sigs = self._signature(x)
        sigs = torch.reshape(sigs, (batch, num_segments, -1))

        if self._include_startpoint:
            x = torch.reshape(x, (batch, num_segments, segment_len, channels))
            sigs = torch.cat([sigs, x[:, :, 0, :].clone()], dim=-1)

        sigs = sigs.transpose(1, 2)
        return sigs
