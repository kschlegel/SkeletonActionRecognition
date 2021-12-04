import torch

from .basetransformer import BaseTransformer


class SpatialTransformer(BaseTransformer):
    """
    Transformer viewing the landmarks of each frame as a sequence in space.

    Takes a tensor of shape
        (batch, in_channels, frames, landmark)
    and considers the landmarks of each frame as a sequene in space, i.e.
    (landmark1_t1,landmark2_t1,....),(landmark1_t2,landmark2_t2,...)
    and computes standard transformer self-attention on each sequence, i.e. how
    each individual landmark in a given frame relates to a given
    landmark in the same frame. This is achieved by moving the frame dimension
    into the batch dimension, applying the standard transfomer module and
    transforming the output back to include the frame dimension.
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
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (batch * frames, in_channels, nodes))
        x = super().forward(x)
        x = torch.reshape(x, (batch, frames, -1, nodes))
        x = x.permute(0, 2, 1, 3)
        return x
