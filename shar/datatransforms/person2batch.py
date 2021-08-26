from typing import Callable

import torch


class Person2Batch(torch.nn.Module):
    """
    Move the person dimension into the batch dimension.

    To allow handling of multiple persons per sequence move the person
    dimension into the batch dimension to apply the model to each person
    individually. After processing of the model we can reshape the array to
    extract the persons back out of the batch dimension and compute the mean
    output of all persons per example for the final output.
    """
    def __init__(self,
                 person_dimension: int,
                 num_persons: int,
                 aggregation: str = "mean") -> None:
        """
        Parameters
        ----------
        person_dimension : int
            Dimension in the input sequence where the person is indexed
        num_persons : int
            Number of persons in each input sequence
        aggregation : str, optional (default is 'mean')
            One of ('mean', 'max') - Method to aggregate results of all persons
            of individual samples.
        """
        super().__init__()
        if person_dimension == 1:
            self.permutation = None
        else:
            self.permutation = (
                [0, person_dimension] +
                list(i for i in range(1, 5) if i != person_dimension))
        self.num_persons = num_persons

        self.aggregation: Callable
        if aggregation == "mean":
            self.aggregation = torch.mean
        elif aggregation == "max":
            self.aggregation = torch.max
        else:
            raise ValueError("Invalid aggregation method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Moves the person dimension into the batch dimension.

        Reorders the tensor for the person dimension to be next to the batch
        dimension and then merges the two dimensions.

        Parameters
        ----------
        x : tensor
            Input tensor of shape
                (batch, [?], in_channels, [?], frames, [?], nodes, [?])
            where one (and only one) of the [?] denotes the person dimension.

        Returns
        -------
        x : tensor
            Output tensor of shape
                (batch * num_persons, out_channels, frames, nodes)
        """
        if self.permutation is not None:
            x = x.permute(*self.permutation).contiguous()
        s = x.size()
        x = x.view(s[0] * s[1], *s[2:])
        return x

    def extract_persons(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the persons out of the batch dimension and aggregates outputs.

        When processing of the main model is done extract the individual
        persons back out of the batch dimension and then aggregate the outputs
        of each sequences persons for a single output per example.

        Parameters
        ----------
        x : tensor
            Input tensor of shape
                (batch, ...)
            where the the batch is composed of entries for individual people
            for individual examples from the original batch.

        Returns
        -------
        x : tensor
            Output tensor of shape
                (batch, ...)
            where batch is now the original batch size, i.e. one entry per
            example and the rest of the tensor has the same shape as the
            parameter had. Each entry is now an aggregation of the results of
            the individual people of each example.
        """
        s = x.size()
        x = x.view((s[0] // self.num_persons, self.num_persons) + s[1:])
        x = self.aggregation(x, dim=1)
        return x
