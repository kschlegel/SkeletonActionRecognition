import torch

from .graphresidual import GraphResidual


class TemporalGraphConvolution(torch.nn.Module):
    """
    Applies a temporal graph convolution over an input graph sequence.

    In the forward pass takes a tensor of shape
        (batch, in_channels, frames, nodes)
    and performs standard convolution over the time dimension. Optionally
    includes a batch norm and residual connection.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 batch_norm: bool = True,
                 residual: bool = False,
                 nonlinearity: bool = True,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels per node
        out_channels : int
            The number of output channels per node
        kernel_size : int
            The size of the kernel of the temporal convolution. Must be an odd
            number (to extend equally into the past and future).
        stride : int, optional (default is 1)
            The stride of the temporal convolution
        batch_norm : bool, optional (default is True)
            Whether to apply batch norm after the convolution
        residual : bool, optional (default is True)
            Whether to include a residual connection around the temporal graph
            convolution
        nonlinearity : bool, optional (default is True)
            If True a ReLU activation is applied before returning the output of
            the convolution
        """
        super().__init__()

        assert kernel_size % 2 == 1
        temporal_padding = ((kernel_size - 1) // 2, 0)

        self.conv = torch.nn.Conv2d(
            out_channels,
            out_channels,
            (kernel_size, 1),
            (stride, 1),
            temporal_padding,
        )

        if batch_norm:
            self.batchnorm = None
        else:
            self.batchnorm = torch.nn.BatchNorm2d(out_channels)

        if not residual:
            self.residual = None
        else:
            self.residual = GraphResidual(in_channels,
                                          out_channels,
                                          stride=stride)

        self.nonlinearity = nonlinearity

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
        y = self.conv(x)

        if self.batchnorm is not None:
            y = self.batchnorm(y)

        if self.residual is not None:
            y += self.residual(x)

        if self.nonlinearity:
            y = torch.nn.functional.relu(y)
        return y
