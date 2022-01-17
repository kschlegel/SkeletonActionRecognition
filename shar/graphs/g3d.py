from typing import Optional

import torch

from .graph import Graph
from .graphresidual import GraphResidual


class G3DGraphConvolution(torch.nn.Module):
    """
    The G3D graph convolution over an input graph sequence.

    In the forward pass takes a tensor of shape
        (batch, in_channels, frames, nodes)
    and performs a G3D graph convolution according to the adjacency matrix
    A of self.graph (the graph provided to the constructor). The G3D graph
    convolution adds cross-spacetime skip connections, i.e. every for every
    node j that is connected to a node i also all instances of node j within a
    temporal window extending either side are directly connected to i.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temporal_window_size: int,
                 graph: Optional[Graph] = None,
                 temporal_window_stride: int = 1,
                 temporal_window_dilation: int = 1,
                 bias: bool = True,
                 batch_norm: bool = True,
                 residual: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels per node
        out_channels : int
            The number of output channels per node
        temporal_window_size : int
            Number of frames to form the sliding window in the time dimension.
        graph : Graph instance, optional (default is None)
            Optionally pass in a graph instance to operate on to allow sharing
            graphs between multiple convolutions.
            If not given a graph instance will be created using the
            graph_options.
        temporal_window_stride : int, optional (Default is 1)
            Stride of the sliding window in the time dimension.
        temporal_window_dilation : int
            Dilation of the sliding window in the time dimension.
        bias : bool, optional (default is True)
            Whether to include bias term in convolution
        batch_norm : bool, optional (default is True)
            Whether to apply batch norm after the convolution
        residual : bool, optional (default is True)
            Whether to add a residual connection around the convolution block
        kwargs :
            kwargs are directly passed through to Graph constructor when
            creating the graph for this convolutional layer (when no graph
            object was passed in). See Graph class constructor for list of
            possible options and values.
        """
        super().__init__()

        if graph is None:
            self.graph = Graph(in_channels=in_channels,
                               out_channels=out_channels,
                               **kwargs)
        else:
            self.graph = graph

        self._temporal_window_size = temporal_window_size
        padding = (temporal_window_size + (temporal_window_size - 1) *
                   (temporal_window_dilation - 1) - 1) // 2
        self.unfold = torch.nn.Unfold(kernel_size=(temporal_window_size, 1),
                                      dilation=(temporal_window_dilation, 1),
                                      stride=(temporal_window_stride, 1),
                                      padding=(padding, 0))
        # Partition strategies with multiple subsets have several instances of
        # adjacency matrices for the different subsets. For this we'll generate
        # a set of features for each subset i.e. an output of shape
        # (batch, subset, out_channels, frames, nodes)
        # This is achieved by outputting subsets*out_channels channels in the
        # convolution and reshaping
        self.num_subsets = self.graph.A.size(0)
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels * self.num_subsets,
                                    kernel_size=(1, 1),
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
                                          stride=temporal_window_stride)

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

        # tile prepends 1s to dims arg for subset(and batch) dimension
        A = torch.tile(A, dims=(self._temporal_window_size, 1))
        # A was already normalised based on number of connections in and out of
        # nodes. Scale proportional to tiling to preserve scaling
        A *= 1 / self._temporal_window_size

        y = self.conv(x)
        batch, subsets_x_channels, frames, nodes = y.size()
        y = self.unfold(y)
        # y = (batch, subsets_x_channels*window_size, num_windows*nodes)
        y = y.view(batch, self.num_subsets,
                   subsets_x_channels // self.num_subsets,
                   self._temporal_window_size, -1, nodes)
        y = y.permute(0, 1, 2, 4, 3, 5)
        y = torch.flatten(y, start_dim=-2)
        # y=(batch,num_subsets,subsets_x_channels,num_windows,window_size*nodes)
        # i.e. same shape as a normal graph convolution but the node dimension
        # now includes the nodes of an entire temporal window

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

        return torch.nn.functional.relu(y)
