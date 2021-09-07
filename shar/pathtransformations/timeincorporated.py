import torch


class TimeIncorporatedTransform(torch.nn.Module):
    """
    Compute the time incorporated transformation of the path.

    Takes a tensor of shape (batch, channels, frames, nodes).
    For each batch element and each node transforms the temporal path of
    channels and length frames, adding an extra channel with a monotone
    increasing time value (integer 0 to num_frames or float 0 to 1).

    Example:
    The path 2,8,4 turns into
        (2,0),(8,1),(4,2)
    """
    def __init__(self, normalised: bool = True) -> None:
        """
        Parameters
        ----------
        normalised: bool, optional (default is True)
            If True time value is floating point value between 0 and 1, if
            False time is integer frame counter.
        """
        super().__init__()
        self.normalised = normalised

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, nodes)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, in_channels+1, frames, nodes)
        """
        batch, channels, frames, nodes = x.size()

        if self.normalised:
            end = 1
            step = 1 / frames
        else:
            end = frames
            step = 1

        time_dimension = torch.arange(start=0,
                                      end=end,
                                      step=step,
                                      dtype=x.dtype,
                                      device=x.device)
        time_dimension = time_dimension.repeat((batch, 1, nodes))
        time_dimension = torch.reshape(time_dimension,
                                       (batch, 1, nodes, frames))
        time_dimension = time_dimension.transpose(2, 3)

        return torch.cat((x, time_dimension), dim=1)
