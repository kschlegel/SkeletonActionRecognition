import torch


class InvisibilityResetTransform(torch.nn.Module):
    """
    Computes the invisibility reset transformation of the path.

    Takes a tensor of shape (batch, channels, frames, nodes).
    For each node transforms the temporal path of each node, adding a
    visibility dimension and two extra time steps. The visibility coordinate is
    set to 1 for any original step of the path, and 0 for the two new steps.
    The first of the two new steps copys the last step of the original path,
    the second one is equal to zero.

    Example:
    The Path 1,2,3 turns into
        (1,1),(2,1),(3,1),(3,0),(0,0)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, nodes)

        Returns
        -------
        x : tensor
            Output tensor of shape (batch, in_channels + 1, frames + 2, nodes)
        """
        batch, channels, frames, nodes = x.size()

        # Add two extra timesteps to the sequence, first a copy of the last
        # timestep and second a constant zero
        y = torch.cat(
            (x, torch.unsqueeze(x[:, :, -1, :], 2),
             torch.zeros(
                 (batch, channels, 1, nodes), dtype=x.dtype, device=x.device)),
            dim=2)

        # Visibility dimension, ones for original path, zero for new timesteps
        visibility = torch.ones((batch, 1, frames + 2, nodes),
                                dtype=x.dtype,
                                device=x.device)
        visibility[:, :, -2:, :] = 0

        y = torch.cat((y, visibility), dim=1)
        return y
