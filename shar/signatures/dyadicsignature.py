import torch
import signatory


class DyadicSignature(torch.nn.Module):
    """
    Computes attention-weighted signatures accross dyadic levels.

    Takes an input sequence of shape
        (batch, in_channels, frames)
    Computes the signatures of all individual segments (in time) at dyadic
    levels (i.e. signatures of the halves of the sequence, signatures of
    quarters of the sequence, etc). At each dyadic level the signatures of all
    segments at that level are combined as a weighted sum based on an attention
    score. The attention score is computed as a linear function of the segments
    signatures. Returns a tensor of shape
        (batch, dyadic_levels, signature_channels)
    """
    def __init__(self, in_channels: int, dyadic_level: int,
                 signature_level: int) -> None:
        """
        in_channels : int
            Number of input channels per frame
        dyadic_level : int
            Compute all dyadic levels up to this level, e.g. dyadic_level=2
            will compute signatures for the whole interval, its halves and its
            quarters.
        signature_level : int
            Truncation level of the signature
        """
        super().__init__()

        self.max_dyadic_level = dyadic_level
        self.signature_level = signature_level

        self.signature = signatory.Signature(depth=signature_level)
        self.attention = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=signatory.signature_channels(
                in_channels, signature_level),
                            out_channels=1,
                            kernel_size=1) for i in range(dyadic_level)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames)

        Returns
        -------
        x : tensor
            One signature per dyadic level as a tensor of shape
            (batch, dyadic_levels, signature_channels)
        """
        batch, channels, frames = x.size()
        y = x.transpose(2, 1)
        # Compute highest dyadic level (i.e. shortest segments) first,
        # signatures of lower dyadic levels can then be computed by
        # successively combining the individual signatures
        num_segments = 2**self.max_dyadic_level
        # Pad with zeros to make sequence length divisible by #segments
        pad = (num_segments - frames % num_segments) % num_segments
        y = torch.cat(
            (y,
             torch.zeros(
                 (batch, pad, channels), device=y.device, dtype=y.dtype)),
            dim=1)
        # Shift the segments into the batch dimension and compute signatures of
        # all segments at once
        y = torch.reshape(
            y, (batch * num_segments, y.shape[1] // num_segments, channels))
        signatures = self.signature(y)
        signatures = torch.reshape(signatures, (batch, num_segments, -1))
        # Compute attention-combined signature for this level
        level_signatures = [
            self.signature_attention(signatures, self.max_dyadic_level)
        ]

        # Successively compute lower dyadic levels (increasing segment length)
        # by combining adjacent signatures
        for dy_lvl in range(self.max_dyadic_level - 1, -1, -1):
            num_segments = 2**dy_lvl
            # split the number of segments in half and move this dyadics level
            # number of segments into the batch dimension. This means that the
            # segments dimension is of size 2, containing neighbouring segments
            # to combine
            signatures = torch.reshape(
                signatures, (batch * num_segments, 2, signatures.shape[2]))
            signatures = signatory.signature_combine(signatures[:, 0],
                                                     signatures[:,
                                                                1], channels,
                                                     self.signature_level)
            signatures = torch.reshape(
                signatures, (batch, num_segments, signatures.shape[1]))
            # signatures is now a tensor containing the dyadic signatures one
            # level up -> compute attention-based combination for this level
            level_signatures.insert(
                0, self.signature_attention(signatures, dy_lvl))

        # Return a stack of the signatures from individual levels
        return torch.stack(level_signatures, dim=1)

    def signature_attention(self, signatures, dyadic_lvl):
        """
        Computes an attention-weighted sum of signatures of segments.

        Computes an attention score for each segments signature as the softmax
        of a linear function of all segment signatures.

        Parameters
        ----------
        signatures : tensor
            Collection of signatures of one dyadic level of shape
            (batch,segments,signature_channels)
        dyadic_lvl : int
            Current dyadic level for which to combine the signatures

        Returns
        -------
        combined_signature : tensor
            Attention-based linear combination of the signatures of the
            individual segments, of shape (batch, signature_channels)
        """
        if dyadic_lvl == 0:
            return torch.squeeze(signatures, dim=1)

        signatures = signatures.transpose(1, 2)
        # Compute a linear map of each signature and softmax the scores
        attention_scores = self.attention[dyadic_lvl - 1](signatures)
        # (batch, 1, segments)
        attention_scores = torch.nn.functional.softmax(attention_scores,
                                                       dim=-1)

        # attention-weighted sum of signatures of the segments
        combined_signature = torch.matmul(attention_scores,
                                          signatures.transpose(1, 2))
        # (batch, signature_channels)
        return torch.squeeze(combined_signature, dim=1)
