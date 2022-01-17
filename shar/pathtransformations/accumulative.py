import torch


class AccumulativeTransform(torch.nn.Module):
    """
    Computes accumulative partial sums of the input sequence.

    Takes a tensor os shape (batch, channels, frames) or
    (batch, channels, frames, streams) as input and computes the partial sums
    along the time dimension (i.e. y_t = sum_{0<=s<=t} x_s). In combination
    with the log-signature this helps to extract the quadratic variation and
    other higher order statistics of the input sequence.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, channels, frames) or
            (batch, channels, frames, streams)

        Returns
        -------
        x : tensor
            Output tensor of the same shape as the input tensor.
            The value at each frame t is the sum of all frames in the input up
            to including t.
        """
        partial_sums = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        partial_sums[:, :, 0] = x[:, :, 0]
        for i in range(1, x.shape[2]):
            partial_sums[:, :, i] = partial_sums[:, :, i - 1] + x[:, :, i]
        return partial_sums
