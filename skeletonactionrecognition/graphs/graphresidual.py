import torch


class GraphResidual(torch.nn.Module):
    """
    Implements a residual connection around a graph convolution block.

    Takes in the input and output channels of the convolution and its stride
    and remaps the input into the right shape using a 1x1 convolution if
    necessary, otw defaults to pure residual.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels per node
        out_channels : int
            The number of output channels per node
        stride : int
            The stride used in the graph convolution
        """
        super().__init__()

        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            # if in_channels != out_channels or temporal stride != 1 then the
            # output of the module is not of the same size as the input so we
            # need to manually adjust the channels/strides with a 1x1
            # convolution to combine with the output with the residual
            self.residual = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            stride=(stride, 1))

    def forward(self, x):
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
        return self.residual(x)
