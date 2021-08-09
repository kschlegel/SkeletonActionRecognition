"""
This implementation is strongly based on
https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
"""
from typing import Optional

import torch

from .graph import Graph
from .graphconvolution import GraphConvolution
from .temporalgraphconvolution import TemporalGraphConvolution
from .graphresidual import GraphResidual


class SpatioTemporalGraphConvolution(torch.nn.Module):
    """
    Applies a spatio-temporal graph convolution over an input graph sequence.

    In the forward pass takes a tensor of shape
        (batch, in_channels, frames, nodes)
    and performs first a standard graph convolution according to the adjacency
    matrix A of self.graph (the graph provided to the constructor) followed by
    a temporal convolution. Optionally includes a residual connection.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temporal_kernel_size: int,
                 graph: Optional[Graph] = None,
                 graph_options: Optional[dict] = {},
                 temporal_stride: int = 1,
                 dropout_p: float = 0,
                 edge_importance_weighting: bool = False,
                 residual: bool = True,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels per node
        out_channels : int
            The number of output channels per node
        temporal_kernel_size : int
            The size of the kernel of the temporal convolution. Must be an odd
            number (to extend equally into the past and future).
        graph : Graph instance, optional (default is None)
            Optionally pass in a graph instance to operate on to allow sharing
            graphs between multiple convolutions.
            If not given a graph instance will be created using the
            graph_options.
        graph_options : dict
            Dictionary of options to be used as parameters when creating the
            graph for this convolutional layer.
            Ignored if graph instance was passed in.
        temporal_stride : int, optional (default is 1)
            The stride of the temporal convolution
        dropout_p : float, optional (default is 0)
            Probability of dropping a node for dropout layer
        edge_importance_weighting : bool, optional (default is False)
            Whether to include a learnable importance weighting to the edges of
            the graph in the graph convolution
        residual : bool, optional (default is True)
            Whether to include a residual connection around the full
            spatiotemporal graph convolution
        """
        super().__init__()

        #assert temporal_kernel_size % 2 == 1
        #temporal_padding = ((temporal_kernel_size - 1) // 2, 0)

        self.gcn = GraphConvolution(
            in_channels,
            out_channels,
            graph=graph,
            graph_options=graph_options,
            edge_importance_weighting=edge_importance_weighting,
            batch_norm=True,
            residual=False)

        self.tcn = TemporalGraphConvolution(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=temporal_kernel_size,
                                            stride=temporal_stride)
        # torch.nn.Sequential(
        #     torch.nn.Conv2d(
        #         out_channels,
        #         out_channels,
        #         (temporal_kernel_size, 1),
        #         (temporal_stride, 1),
        #         temporal_padding,
        #     ),
        #     torch.nn.BatchNorm2d(out_channels),
        #     torch.nn.Dropout(dropout_p, inplace=True),
        # )

        self.dropout = torch.nn.Dropout(dropout_p, inplace=True)

        if not residual:
            self.residual = None
        else:
            self.residual = GraphResidual(in_channels,
                                          out_channels,
                                          stride=temporal_stride)

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
        y = self.gcn(x)
        y = self.tcn(y)

        y = self.dropout(y)

        if self.residual:
            y += self.residual(x)

        return torch.nn.functional.relu(y)
