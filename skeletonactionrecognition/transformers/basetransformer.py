"""
This implementation generally follows
https://github.com/Chiaraplizz/ST-TR
and
https://github.com/leaderj1001/Attention-Augmented-Conv2d
"""

from typing import Tuple, Union

import torch


class BaseTransformer(torch.nn.Module):
    """
    Transformer base-class operating on a sequence in R^n.

    Takes a tensor of shape
        (batch, in_channels, sequence_items)
    and computes standard transformer self-attention on this sequence, i.e.
    how each of the sequence items relates to a given sequence item.
    Outputs a tensor of shape
        (batch, out_channels, sequence_items)

    Skeletons sequences are sequences in R^l*R^d. We can employ different
    strategies to transform such a sequence back into a space R^n and then pass
    that sequence back up into this base class.

    This is done with
     - SpatioTemporal Transformer: considers each landmark in each frame as a
       sequence element by iterating all of them as
          (landmark1_t1,landmark2_t1,....,landmark1_t2,landmark2_t2,...)
     - Spatial Transformer: considers each frame a sequence in space by moving
          the frame dimension into the batch dimension
     - Temporal Transformer: Considers each landmark a sequence in time by
          moving the joint dimension into the batch dimension
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dim_key: int,
            dim_value: int,
            num_heads: int,
            residual: bool = True,
            normalisation: Union[None, str, int, Tuple[int]] = None) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels per landmark
        out_channels : int
            Number of output channels per landmark
        dim_key : int
            Dimension of the query and key vectors
        dim_value : int
            Dimension of the value vectors
        num_heads : int
            Number of attention heads
        residual : bool, optional (default is True)
            Whether to add a residual connection around the transformer block
        normalisation : str, int or tuple of ints, optional (default is None)
            Normalisation to be applied after the self-attention block (and
            after adding the residual if selected). One of
            (None, 'batch') or an int or tuple of ints - No normalisation is
            used if None. if set to 'batch' standard batch norm is used. If an
            int or tuple of ints is passed in applies layer norm with the value
            used as normalised_shape argument ( see PyTorch LayerNorm docs).
        """
        super().__init__()

        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads
        self.gradient_scaling = dim_key**-0.5

        self.qkv_conv = torch.nn.Conv1d(in_channels,
                                        num_heads * (2 * dim_key + dim_value),
                                        kernel_size=1)
        self.W_o = torch.nn.Conv1d(num_heads * dim_value,
                                   out_channels,
                                   kernel_size=1)

        if not residual:
            self.residual = None
        else:
            self.residual = torch.nn.Conv1d(in_channels,
                                            out_channels,
                                            kernel_size=1)

        if normalisation is None:
            self.normalisation = None
        elif normalisation == "batch":
            self.normalisation = torch.nn.BatchNorm1d(
                num_features=out_channels)
        else:
            self.normalisation = torch.nn.LayerNorm(
                normalised_shape=normalisation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, sequence_items)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, out_channels, sequence_items)
        """
        batch, __, sequence_items = x.size()

        q, k, v = self.compute_qkv(x)
        # q,k,v have shape
        # (batch, num_heads, [dim query/key/value], sequence_items)
        # Compute w = softmax(Q*K^T/sqrt(d_k))
        logits = self.gradient_scaling * torch.matmul(q.transpose(2, 3), k)
        weights = torch.nn.functional.softmax(logits, dim=-1)
        # weights are of shape
        #    (batch, heads, sequence_items, sequence_items)

        # Compute z_i = w*V
        z_i = torch.matmul(weights, v.transpose(2, 3))
        # z_i is of shape (batch, heads, sequence_items, dim_value)
        z_i = z_i.transpose(2, 3)
        # z_i is of shape (batch, heads, dim_value, sequence_items)
        z_i = torch.reshape(
            z_i, (batch, self.num_heads * self.dim_value, sequence_items))
        # z_i is concatenation of outputs of individual heads
        # -> Apply linear function to combine to final transformed output
        z = self.W_o(z_i)

        if self.residual is not None:
            z += self.residual(x)

        if self.normalisation is not None:
            z = self.normalisation(z)

        return z

    def compute_qkv(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, __, sequence_items = x.size()
        qkv = self.qkv_conv(x)
        # qkv has shape
        # (batch, num_heads * (dim_key+dim_query+dim_value), sequence_items)
        # qkv_conv is linear map
        # -> seperate out the heads and split into individual q,k,v elements
        qkv = torch.reshape(qkv, (batch, self.num_heads, -1, sequence_items))
        q, k, v = torch.split(qkv,
                              [self.dim_key, self.dim_key, self.dim_value],
                              dim=2)
        # q,k,v have shape
        # (batch, num_heads, [dim query/key/value], sequence_items)

        # In summary: qkv_conv is a linear map
        #    R^{in} -> R^{num_heads * (dim_key + dim_query + dim_value)}
        # mapping the feature vector of each sequence item to a vector
        #  (query1,key1,value1,query2,key2,...) where the number refers to the
        # attention head.

        return q, k, v
