import torch


class ChannelwiseBatchNorm(torch.nn.Module):
    """
    Batch normalisation on individual channels of nodes.

    In particular this allows to batch normalise each coordinate stream of each
    landmark individually.
    """
    def __init__(self, in_channels: int, landmarks: int) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of channels per landmark
        landmarks : int
            Number of landmarks per frame
        """
        super().__init__()
        self.data_batch_norm = torch.nn.BatchNorm1d(in_channels * landmarks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batch normalises each channel of each node individually.

        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, nodes)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, out_channels, frames, nodes)
        """
        batch, channels, frames, nodes = x.size()
        x = x.transpose(3, 2).contiguous()
        x = x.view(batch, channels * nodes, frames)
        x = self.data_batch_norm(x)
        x = x.view(batch, channels, nodes, frames)
        x = x.transpose(3, 2).contiguous()
        return x
