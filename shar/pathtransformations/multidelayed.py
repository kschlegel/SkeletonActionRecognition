import torch


class MultiDelayedTransform(torch.nn.Module):
    """
    Compute the multi-delayed transformation of a path.

    Expects a tensor of shape (batch, channels, frames, nodes). This is a
    variant of the lead-lag transformation which instead of advancing time one
    component at a time it advances time by one in each component every step so
    that each element of the multi-delayed path contains the last delay
    elements of the original path. Pads with zeros at the ends.

    Example:
    The path 1,2,3 with delay 1 turns into
        (1,0),(2,1),(3,2),(0,3)
    """
    def __init__(self, delay: int = 1) -> None:
        """
        Parameters
        ----------
        delay : int
            Number of timesteps up to which to delay the path
        """
        super().__init__()
        self.delay = delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, nodes)

        Returns
        -------
        x : tensor
            Output tensor of shape
              (batch, (in_channels * (delay + 1)), frames + delay, nodes)
        """
        batch, channels, frames, nodes = x.size()

        # Empty target array with an extra dimension for each delay
        y = torch.zeros(
            (batch, self.delay + 1, channels, frames + self.delay, nodes),
            dtype=x.dtype,
            device=x.device)

        # Copy the delayed paths into the target array
        for i in range(self.delay + 1):
            y[:, i, :, i:i + frames, :] = x

        # Flatten the delay into the channels so that a n dimensional input
        # path results in a n*(delay+1) dimensional output path
        y = torch.reshape(y, (batch, channels *
                              (self.delay + 1), frames + self.delay, nodes))
        return y
