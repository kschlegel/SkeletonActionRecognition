import math

import torch

from .segmentsignatures import _SegmentSignatures


class _SegmentSignaturesSequential(_SegmentSignatures):
    """
    Compute the segment-wise signatures or log-signatures sequentially.

    See _SegmentSignatures for detailed description.

    This version computes the segments signatures sequentially in a loop.
    Instead of moving the segment dimension into the batch dimension for
    parallel computation. This does not require padding at the end but the last
    interval may be shorter than the other segments.
    """
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

        # Set number of segments &segment length values and segment bdry pts
        # depending on settings
        if self._num_segments is not None:
            num_segments = self._num_segments
            segment_len = int(math.ceil(frames / num_segments))
        elif self._segment_len is not None:
            segment_len = self._segment_len
            num_segments = int(math.ceil(frames / segment_len))
        else:
            raise ValueError("Either number of segments or segment length "
                             "must be specified.")
        segment_pts = [i * segment_len for i in range(num_segments + 1)]
        segment_pts[-1] = frames

        # Iteratively compute segment signature
        sig_collection = []
        for i in range(num_segments):
            sig_collection.append(
                self._signature(x[:, segment_pts[i]:segment_pts[i + 1], :]))
        sigs = torch.stack(sig_collection, dim=1)

        if self._include_startpoint:
            sigs = torch.cat([sigs, x[:, segment_pts[:-1]].clone()], dim=-1)

        sigs = sigs.transpose(1, 2)
        return sigs
