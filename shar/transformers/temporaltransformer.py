import torch

from .basetransformer import BaseTransformer


class TemporalTransformer(BaseTransformer):
    """
    Transformer viewing each landmark as a sequence in time.

    Takes a tensor of shape
        (batch, in_channels, frames, landmark)
    and considers each landmark as a sequene in time, i.e.
    (landmark1_t1,landmark1_t2,....),(landmark2_t1,landmark2_t2,...)
    and computes standard transformer self-attention on each sequence, i.e. how
    a given landmark in each individual frame relates to the same landmark in a
    given frame. This is achieved by moving the landmark dimension into the
    batch dimension, applying the standard transfomer module and transforming
    the output back to include the landmark dimension.
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
        x = x.permute(0, 3, 1, 2)
        x = torch.reshape(x, (batch * nodes, in_channels, frames))
        x = super().forward(x)
        x = torch.reshape(x, (batch, nodes, -1, frames))
        x = x.permute(0, 2, 3, 1)
        return x
