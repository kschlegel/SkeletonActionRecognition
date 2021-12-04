from typing import Optional

import torch

from .graph import Graph
from .graphconvolution import GraphConvolution
from .temporalgraphconvolution import TemporalGraphConvolution
from .graphresidual import GraphResidual
from .graphlayouts import GraphLayout
from .graphoptions import GraphOptions


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
                 graph_layout: Optional[GraphLayout] = None,
                 graph_options: Optional[GraphOptions] = None,
                 temporal_stride: int = 1,
                 dropout_p: float = 0,
                 residual: bool = True,
                 nonlinearity: bool = True,
                 spatial_nonlinearity: bool = True,
                 temporal_nonlinearity: bool = True,
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
            If not given a graph instance will be created.
        graph_layout : GraphLayout object, optional (default is None)
            GraphLayout object defining the connections and the center node of
            the graph to be created. Ignored if graph instance was passed in.
            Must be specified if graph is not given.
        graph_options : GraphOptions object, optional (default is None)
            GraphOptions object defining various properties of the
            graph to be created, such as components of the adjacency matrix and
            normalisation method. See GraphOptions documentation for more
            information. Ignored if graph instance was passed in. Must be
            specified if graph is not given.
        temporal_stride : int, optional (default is 1)
            The stride of the temporal convolution
        dropout_p : float, optional (default is 0)
            Probability of dropping a node for dropout layer
        residual : bool, optional (default is True)
            Whether to include a residual connection around the full
            spatiotemporal graph convolution
        nonlinearity : bool, optional (default is True)
            If True a ReLU activation is applied before returning the output of
            the convolution
        spatial_nonlinearity : bool, optional (default is True)
            If True a ReLU activation is applied after the spatial convolution
        temporal_nonlinearity : bool, optional (default is True)
            If True a ReLU activation is applied after the temporal convolution
        """
        super().__init__()

        self.gcn = GraphConvolution(in_channels,
                                    out_channels,
                                    graph=graph,
                                    graph_layout=graph_layout,
                                    graph_options=graph_options,
                                    batch_norm=True,
                                    residual=False,
                                    **kwargs)

        self.tcn = TemporalGraphConvolution(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=temporal_kernel_size,
                                            stride=temporal_stride)

        self.dropout = torch.nn.Dropout(dropout_p, inplace=True)

        if not residual:
            self.residual = None
        else:
            self.residual = GraphResidual(in_channels,
                                          out_channels,
                                          stride=temporal_stride)

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
        y = self.gcn(x)
        y = self.tcn(y)

        y = self.dropout(y)

        if self.residual:
            y += self.residual(x)

        if self.nonlinearity:
            y = torch.nn.functional.relu(y)
        return y
