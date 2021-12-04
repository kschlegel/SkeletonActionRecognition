from typing import Optional

import torch

from .graph import Graph
from .graphlayouts import GraphLayout
from .graphoptions import GraphOptions
from .graphresidual import GraphResidual


class GraphConvolution(torch.nn.Module):
    """
    The basic graph convolution over an input graph sequence.

    In the forward pass takes a tensor of shape
        (batch, in_channels, frames, nodes)
    and performs a standard graph convolution according to the adjacency matrix
    A of self.graph (the graph provided to the constructor)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 graph: Optional[Graph] = None,
                 graph_layout: Optional[GraphLayout] = None,
                 graph_options: Optional[GraphOptions] = None,
                 temporal_kernel_size: int = 1,
                 temporal_stride: int = 1,
                 temporal_padding: int = 0,
                 temporal_dilation: int = 1,
                 bias: bool = True,
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
        temporal_kernel_size : int, optional (default is 1)
            Size of the convolutional kernel in the frame dimension.
        temporal_stride : int, optional (default is 1)
            Tride of the convolutional kernel in the frame dimension.
        temporal_padding : int, optional (default is 0)
            Padding of the convolutional kernel in the frame dimension.
        temporal_dilation : int, optional (default is 1)
            Dilation of the convolutional kernel in the frame dimension.
        bias : bool, optional (default is True)
            Whether to include bias term in convolution
        batch_norm : bool, optional (default is True)
            Whether to apply batch norm after the convolution
        residual : bool, optional (default is True)
            Whether to add a residual connection around the convolution block
        nonlinearity : bool, optional (default is True)
            If True a ReLU activation is applied before returning the output of
            the convolution
        """
        super().__init__()

        if graph is None:
            if graph_layout is None:
                raise ValueError("The graph layout needs to be specified when "
                                 "not providing a graph instance")
            if graph_options is None:
                raise ValueError("The graph options need to be specified when "
                                 "not providing a graph instance")
            self.graph = Graph(graph_layout=graph_layout,
                               graph_options=graph_options,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               **kwargs)
        else:
            self.graph = graph
        # Partition strategies with multiple subsets have several instances of
        # adjacency matrices for the different subsets. For this we'll generate
        # a set of features for each subset i.e. an output of shape
        # (batch, subset, out_channels, frames, nodes)
        # This is achieved by outputting subsets*out_channels channels in the
        # convolution and reshaping
        self.num_subsets = self.graph.A.size(0)
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels * self.num_subsets,
                                    kernel_size=(temporal_kernel_size, 1),
                                    padding=(temporal_padding, 0),
                                    stride=(temporal_stride, 1),
                                    dilation=(temporal_dilation, 1),
                                    bias=bias)

        if not batch_norm:
            self.batch_norm = None
        else:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)

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
        A = self.graph(x)

        y = self.conv(x)
        batch, subsets_channels, frames, nodes = y.size()
        y = y.view(batch, self.num_subsets,
                   subsets_channels // self.num_subsets, frames, nodes)

        # b=batch, s=subset, c=channels, f=frames n=m=nodes(A is square)
        # The einsum results in
        #    x^{out}_{b,c,f,m} = \sum_s \sum_n x^{in}_{b,s,c,f,n} * A_{s,n,m}
        # \sum_s sums over the different partial adjacency matrices which arise
        # in partition strategies with multiple subsets i.e. the sum collapses
        # when partition_strategy='uniform'
        # \sum_n computes the sampling function by multiplying the feature
        # vector by the column of A describing which nodes are connected, i.e.
        # influence the output at the given node. All others are multiplied by
        # the 0 in A. Moreover it performs normalisation with respect to the
        # subgroup cardinality since A is normalised in this way.
        if len(A.shape) == 3:
            A_shape = "snm"
        elif len(A.shape) == 4:
            # If a data dependent component is used in the adjacency matrix
            # then we have one matrix per batch element and per subset
            A_shape = "bsnm"
        else:
            raise Exception("Invalid adjacency matrix shape")
        y = torch.einsum('bscfn,' + A_shape + '->bcfm', (y, A)).contiguous()

        if self.batch_norm is not None:
            y = self.batch_norm(y)

        if self.residual is not None:
            y += self.residual(x)

        if self.nonlinearity:
            y = torch.nn.functional.relu(y)
        return y
