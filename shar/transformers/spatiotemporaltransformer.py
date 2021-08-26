import torch

from .basetransformer import BaseTransformer


class SpatioTemporalTransformer(BaseTransformer):
    """
    Transformer viewing the input as sequence of landmarks and frames.

    Takes a tensor of shape
        (batch, in_channels, frames, landmark)
    and considers it as a sequene of landmarks and frames, i.e.
    (landmark1_t1,landmark2_t1,....,landmark1_t2,landmark2_t2,...)
    computes standard transformer self-attention on this sequence, i.e. how
    each individual landmark in each individual frame relates to a given
    landmark in a given frame.
    Outputs a tensor of shape
        (batch, out_channels, frames, landmark)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, nodes)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, out_channels, frames, nodes)
        """
        batch, in_channels, frames, nodes = x.size()
        x = torch.reshape(x, (batch, in_channels, frames * nodes))
        x = super().forward(x)
        x = torch.reshape(x, (batch, -1, frames, nodes))
        return x
