from typing import Tuple

import torch


class Transformer(torch.nn.Module):
    """
    Transformer viewing the input as sequence of landmarks and frames.

    Takes a tensor of shape
        (batch, in_channels, frames, landmark)
    and considers it as a sequene of landmarks and frames, i.e.
    (landmark1_t1,landmark2_t1,....,landmark1_t2,landmark2_t2,...)
    computes standard transformer self-attention on this sequence, i.e. how
    each individual landmark in each individual frame relates to a given
    landmark in a given frame.
    Outputs a tensor of shape
        (batch, out_channels, frames, landmark)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dim_key: int,
                 dim_value: int,
                 num_heads: int,
                 residual: bool = True) -> None:
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
        """
        super().__init__()

        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads
        self.gradient_scaling = dim_key**-0.5

        self.qkv_conv = torch.nn.Conv2d(in_channels,
                                        num_heads * (2 * dim_key + dim_value),
                                        kernel_size=1)
        self.W_o = torch.nn.Conv2d(num_heads * dim_value,
                                   out_channels,
                                   kernel_size=1)

        if not residual:
            self.residual = None
        else:
            self.residual = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, landmarks)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, out_channels, frames, landmarks)
        """
        batch, __, frames, landmarks = x.size()

        q, k, v = self.compute_qkv(x)
        # Compute w = softmax(Q*K^T/sqrt(d_k))
        logits = self.gradient_scaling * torch.matmul(q.transpose(2, 3), k)
        weights = torch.nn.functional.softmax(logits, dim=-1)
        # weights are of shape
        #    (batch, heads, frames*landmarks, frames*landmarks)

        # Compute z_i = w*V
        z_i = torch.matmul(weights, v.transpose(2, 3))
        # z_i is of shape (batch, heads, frames*landmarks, dim_value)
        z_i = z_i.transpose(2, 3)
        # z_i is of shape (batch, heads, dim_value, frames*landmarks)
        z_i = torch.reshape(
            z_i, (batch, self.num_heads * self.dim_value, frames, landmarks))

        # z_i is concatenation of outputs of individual heads
        # -> Apply linear function to combine to final transformed output
        z = self.W_o(z_i)

        if self.residual is not None:
            z += self.residual(x)

        return z

    def compute_qkv(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, __, frames, landmarks = x.size()
        qkv = self.qkv_conv(x)
        # qkv has shape
        # (batch, num_heads * (dim_key+dim_query+dim_value), frames, landmarks)
        # qkv_conv is linear map
        # -> seperate out the heads and split into individual q,k,v elements
        qkv = torch.reshape(qkv,
                            (batch, self.num_heads, -1, frames * landmarks))
        q, k, v = torch.split(qkv,
                              [self.dim_key, self.dim_key, self.dim_value],
                              dim=2)
        # q,k,v have shape
        # (batch, num_heads, [dim query/key/value], frames*landmarks)

        # In summary: qkv_conv is a linear map
        #    R^{in} -> R^{num_heads * (dim_key + dim_query + dim_value)}
        # mapping the feature vector of each landmark in each frame to a vector
        #  (query1,key1,value1,query2,key2,...) where the number refers to the
        # attention head.

        return q, k, v
