import torch


class EmbeddingLayer(torch.nn.Module):
    """
    Computes a low-dimensional embedding of a spatio-temporal path.
    """
    def __init__(self, in_channels, out_channels, landmarks):
        super().__init__()
        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=32,
                            kernel_size=1),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=16,
                            kernel_size=(3, 1),
                            padding=(1, 0)))

        self.fully_connected = torch.nn.Conv1d(in_channels=16 * landmarks,
                                               out_channels=out_channels,
                                               kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, streams)

        Returns
        -------
        x : tensor
            Output_tensor of shape (batch, out_channels, frames, 1)
        """
        batch, __, frames = x.shape[:3]
        x = self.convolutions(x)
        x = x.transpose(2, 3)
        x = torch.reshape(x, (batch, -1, frames))
        x = self.fully_connected(x)
        x = x.unsqueeze(-1)
        return x
