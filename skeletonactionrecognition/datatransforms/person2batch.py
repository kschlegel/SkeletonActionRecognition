import torch


class Person2Batch(torch.nn.Module):
    """
    Move the person dimension into the batch dimension.

    To allow handling of multiple persons per sequence move the person
    dimension into the batch dimension to apply the model to each person
    individually. After processing of the model we can reshape the array to
    extract the persons back out of the batch dimension and compute the mean
    output of all persons per example for the final output.

    TODO: Add an optio nfor other aggregation operations e.g. max
    """
    def __init__(self, person_dimension, num_persons):
        """
        Parameters
        ----------
        person_dimension : int
            Dimension in the input sequence where the person is indexed
        num_persons : int
            Number of persons in each input sequence
        """
        super().__init__()
        self.permutation = (
            [0, person_dimension] +
            list(i for i in range(1, 5) if i != person_dimension))
        self.num_persons = num_persons

    def person2batch(self, x):
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
        x = x.permute(*self.permutation).contiguous()
        s = x.size()
        x = x.view(s[0] * s[1], *s[2:])
        return x

    def extract_persons(self, x):
        """
        Extracts the persons out of the batch dimension and aggregates outputs.

        When processing of the main model is done extract the individual
        persons back out of the batch dimension and then aggregate the outputs
        for a single output per example.
        """
        s = x.size()
        x = x.view(s[0] // self.num_persons, self.num_persons,
                   s[1:]).mean(dim=1)
        return x
