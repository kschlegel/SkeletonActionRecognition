from typing import List, Tuple, Optional

import torch

from .graphlayouts import GraphLayout
from .graphoptions import GraphOptions
from .graphpartitions import (GraphPartition, UniformPartition,
                              DistancePartition, SpatialPartition)


class Graph(torch.nn.Module):
    def __init__(self,
                 graph_layout: GraphLayout,
                 graph_options: GraphOptions,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 embedding_dimension: Optional[int] = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        graph_layout : GraphLayout object
            GraphLayout object defining the connections and the center node of
            the graph to be created. See GraphLayout documentation for more
            information.
        graph_options : GraphOptions object
            GraphOptions object defining various properties of the
            graph to be created, such as components of the adjacency matrix and
            normalisation method. See GraphOptions documentation for more
            information.
        in_channels : int
            Dimension of the data at each vertex.
            Only needs to be supplied when using a data-dependent component in
            the adjacency matrix.
        out_channels : int
            Dimension of the data at each vertex after applying graph conv.
            Only needs to be supplied when using a data-dependent component in
            the adjacency matrix and if embedding_dimension is not supplied. In
            this case embedding_dimension will default to out_channels // 4
        embedding_dimension : int
            Dimension of the data embedding for the computation of the
            data-dependent component of the adjacency matrix.
            Only needs to be supplied when using a data-dependent component in
            the adjacency matrix.
        """
        super().__init__()

        # Static component of the adjacency matrix
        self.A: torch.Tensor
        self.register_buffer(
            "A", self._compute_adjency_matrix(graph_layout, graph_options))

        # Optional learnable component for the adjacency matrix
        self.learnable_adjacency = graph_options.learnable_adjacency
        if self.learnable_adjacency:
            # TODO: param to init this uniformly?
            self.B = torch.nn.Parameter(torch.zeros(self.A.shape))

        # Optional data-dependent component for the adjacency matrix
        self.data_dependent_adjacency = graph_options.data_dependent_adjacency
        if self.data_dependent_adjacency:
            if in_channels is None:
                raise Exception(
                    "The input dimension needs to be specified when using a "
                    "data-dependent component in the adjacency matrix.")
            if embedding_dimension is None:
                if out_channels is None:
                    raise Exception(
                        "The embedding or output dimension need to be "
                        "specified when using a data-dependent component in "
                        "the adjacency matrix.")
                else:
                    self.embedding_dimension = out_channels // 4
            else:
                self.embedding_dimension = embedding_dimension

            self.phi = torch.nn.ModuleList()
            self.theta = torch.nn.ModuleList()
            for i in range(self.A.shape[0]):
                self.phi.append(
                    torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=self.embedding_dimension,
                                    kernel_size=1))
                self.theta.append(
                    torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=self.embedding_dimension,
                                    kernel_size=1))

        # Optional learnable importance weighting for each edge in the graph
        self.edge_importance: Optional[torch.Tensor]
        if graph_options.edge_importance_weighting:
            self.edge_importance = torch.nn.Parameter(torch.ones(
                self.A.size()))
        else:
            self.edge_importance = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assembles and returns the adjacency matrix with all selected components

        If selected Computes the data-dependent part of the adjacency matrix
        and broadcasts the fixed and learnable components onto the batched
        data-dependent component.

        Parameters
        ----------
        x : tensor
            Input tensor of shape (batch, in_channels, frames, nodes)
            Only used if the adjacency matrix has a data dependent component

        Returns
        -------
        A : torch.tensor
            Adjacency matrix of the graph as the sum of all selected components
            from: base component, learnable component, data-dependent component
            If no data-dependent component is used this will be of shape
               (partition, num_nodes, num_nodes)
            With a data-dependent component this will be of shape
               (batch, partition, num_nodes, num_nodes)
        """
        A = self.A
        if self.learnable_adjacency:
            A = A + self.B

        if self.data_dependent_adjacency:
            # update the data-dependent component of the adjacency matrix
            batch, __, frames, nodes = x.size()
            C = torch.zeros((x.shape[0], ) + self.A.shape,
                            device=self.A.device)
            for i in range(self.A.shape[0]):
                C1 = self.phi[i](x).permute(0, 3, 1, 2).contiguous().view(
                    batch, nodes, self.embedding_dimension * frames)
                C2 = self.theta[i](x).view(batch,
                                           self.embedding_dimension * frames,
                                           nodes)
                C[:, i] = torch.nn.functional.softmax(torch.matmul(C1, C2) /
                                                      C1.size(-1),
                                                      dim=-2)
            A = A + C
        if self.edge_importance is not None:
            A = A * self.edge_importance
        return A

    def _compute_adjency_matrix(
        self,
        graph_layout: GraphLayout,
        graph_options: GraphOptions,
    ) -> torch.Tensor:
        """
        Computes the adjacency matrix A for different partitioning strategies.

        When a partition strategy with more than one subgroup is used (i.e.
        anything != uniform), then the adjacency matrix is split into several
        individual matrices for which the output is computed and then summed
        over all matrix instances, i.e. A=sum_j A_j and
        x_out = sum_j A_j*x_in*w
        The returned adjacency matrices are already nornmalised by subset
        cardinalities using the chosen normalisation method.

        Parameters
        ----------
        graph_layout : GraphLayout object
            GraphLayout object defining the connections and the center node of
            the graph to be created. See GraphLayout documentation for more
            information.
        graph_options : GraphOptions object
            GraphOptions object defining various properties of the
            graph to be created, such as components of the adjacency matrix and
            normalisation method. See GraphOptions documentation for more
            information.

        Returns
        -------
        A : tensor
            Set of normalised adjacency matrices (normalised by subset
            cardinalities)
        """
        distance_matrix = self._compute_distance_matrix(
            graph_layout.edges, graph_options.directed_graph)

        distance_steps = list(
            range(0, graph_options.max_neighbour_distance + 1,
                  graph_options.dilation))
        adjacency = torch.zeros(distance_matrix.shape)
        for d in distance_steps:
            adjacency[distance_matrix == d] = 1

        if graph_options.normalise_first:
            adjacency = self._normalise(adjacency,
                                        graph_options.normalisation_method,
                                        graph_options.directed_graph)

        partition_strategy: GraphPartition
        if isinstance(graph_options.partition_strategy, str):
            if graph_options.partition_strategy == 'uniform':
                partition_strategy = UniformPartition()
            elif graph_options.partition_strategy == 'distance':
                partition_strategy = DistancePartition(
                    **graph_options.asdict())
            elif graph_options.partition_strategy == 'spatial':
                partition_strategy = SpatialPartition(**graph_layout.asdict())
            else:
                raise ValueError("Invalid partition strategy")
        else:
            partition_strategy = graph_options.partition_strategy

        adjacency = partition_strategy.apply(adjacency_matrix=adjacency,
                                             distance_matrix=distance_matrix,
                                             distance_steps=distance_steps)

        if not graph_options.normalise_first:
            # Late normalisation means normalising each partition component
            # individually
            for i in range(adjacency.shape[0]):
                adjacency[i] = self._normalise(
                    adjacency[i], graph_options.normalisation_method,
                    graph_options.directed_graph)

        return adjacency

    @staticmethod
    def _compute_distance_matrix(edges: List[Tuple[int, int]],
                                 directed_graph: bool) -> torch.Tensor:
        """
        Computes the matrix of pairwise distances between nodes.

        Returns a matrix of shape (num_nodes, num_nodes) where each entry
        contains the minimal number of steps along edges to get from nodes i to
        node j.

        Parameters
        ----------
        edges: List of integer tuples
            List of pairs of integers, describing the connections within the
            graph, excluding self-connections, which are added automatically.
        directed_graph: bool
            Whether the graph is directed or undirected

        Returns
        -------
        A : tensor
            The matrix of pairwise node distances
        """
        num_nodes = max(max(c) for c in edges) + 1

        edges += [(i, i) for i in range(num_nodes)]

        A = torch.zeros((num_nodes, num_nodes))
        for i, j in edges:
            A[i, j] = 1  # edge from i to j
            if not directed_graph:
                A[j, i] = 1  # edge from j to i

        # Let A_k be the matrix with a_{i,j}=1 if A^k for the adjacency matrix
        # k is positive, nonzero at i,j and 0 otherwise. Then A_k - A_{k-1} is
        # 1 at i,j if and only if node i and j have distance d and 0 everywhere
        # else. Thus we can compute the full distance matrix by successively
        # computing matrix powers of the adjacency matrix A to iteratively
        # determine nodes at the next distance step until all distances have
        # been determined
        distance_matrix = torch.full((num_nodes, num_nodes), float("Inf"))
        B = torch.eye(num_nodes)
        distance_matrix[B.to(torch.bool)] = 0
        i = 0
        while torch.max(distance_matrix) == float("Inf"):
            C = torch.matmul(A, B)
            mask = (torch.minimum(C, torch.tensor(1)) -
                    torch.minimum(B, torch.tensor(1)))
            distance_matrix[mask.to(torch.bool)] = i + 1
            B = C
            i += 1
        return distance_matrix

    def _normalise(self, adjacency: torch.Tensor, normalisation_method: str,
                   directed_graph: bool) -> torch.Tensor:
        """
        Normalises the adjacency matrix by the given normalisation method.

        Parameters
        ----------
        adjacency : tensor
            Adjacency matrix A of the graph of shape (num_nodes, num_nodes)
            with a_{i,j}=1 if nodes i and j have are within the same subgroup,
            0 otw.
        normalisation_method : str
            One of ('mean_pooling', 'symmetric') - String identifying the
            normalisation method applied to the adjacency matrix to normalise
            contributions based on group sizes
        directed_graph: bool
            Whether the graph is directed or undirected

        Returns
        -------
        A_normalised : tensor
            Adjacency matrix with each entry normalised by the cardinality of
            the corresponding subset.
        """
        if normalisation_method == "mean_pooling":
            normalized_adjacency = self._mean_pooling_normalization(adjacency)
        elif normalisation_method == "symmetric":
            normalized_adjacency = self._symmetric_normalization(
                adjacency, directed_graph)
        else:
            raise Exception("Invalid normalisation method.")
        return normalized_adjacency

    @staticmethod
    def _mean_pooling_normalization(A: torch.Tensor) -> torch.Tensor:
        """
        Normalises the adjacency matrix by subset cardinality of the root node.

        In mean pooling updates, the contribution of the relation of the root
        node i to a neighbour node j is normalised by 1/|B_i|, the cardinality
        of the neighbourset of the root node i.

        Parameters
        ----------
        A : tensor
            Adjacency matrix A of the graph of shape (num_nodes, num_nodes)
            with a_{i,j}=1 if nodes i and j have a distance of at most
            max_neighbour_distance and =0 otw.

        Returns
        -------
        A_normalised : tensor
            Adjacency matrix with each entry normalised by the cardinality of
            the corresponding subset.
        """
        # count number of neighbours of node i
        neighbour_counts = torch.sum(A, dim=0)
        normalisation = torch.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)
        for i in range(A.shape[1]):
            if neighbour_counts[i] > 0:
                normalisation[i, i] = neighbour_counts[i]**(-1)
        # normalise at node i by the count of neighbours of i
        A_normalised = torch.matmul(A, normalisation)
        return A_normalised

    @staticmethod
    def _symmetric_normalization(A: torch.Tensor,
                                 directed_graph: bool) -> torch.Tensor:
        """
        Normalises the adjacency matrix by subset cardinalities at both nodes.

        In symmetric normalisation the contribution of the relation of the root
        node i to a neighbour node j is normalised by 1/sqrt(|B_i|*|B_j|), the
        squareroot of the product of the cardinalities of neighboursets of the
        root node i and neighbour node j.

        Parameters
        ----------
        A : tensor
            Adjacency matrix A of the graph of shape (num_nodes, num_nodes)
            with a_{i,j}=1 if nodes i and j have a distance of at most
            max_neighbour_distance and =0 otw.

        Returns
        -------
        A_normalised : tensor
            Adjacency matrix with each entry normalised by the cardinality of
            the corresponding subset.
        """
        # count edges going out of node i
        neighbour_counts = torch.sum(A, dim=0)
        normalisation_source = torch.zeros((A.shape[1], A.shape[1]),
                                           dtype=A.dtype)
        for i in range(A.shape[1]):
            if neighbour_counts[i] > 0:
                normalisation_source[i, i] = neighbour_counts[i]**(-0.5)
            else:
                normalisation_source[i, i] = 1e-6
        if directed_graph:
            # count edges going into node i
            neighbour_counts = torch.sum(A, dim=1)
            normalisation_target = torch.zeros((A.shape[0], A.shape[0]),
                                               dtype=A.dtype)
            for i in range(A.shape[0]):
                if neighbour_counts[i] > 0:
                    normalisation_target[i, i] = neighbour_counts[i]**(-0.5)
                else:
                    normalisation_target[i, i] = 1e-6
        else:
            # undirected graph: #edges going in = #edges going out
            normalisation_target = normalisation_source
        # normalise contribution of edge from i to j by sqrt of number of
        # outgoing edges at i and incoming edges at j
        A_normalised = torch.matmul(torch.matmul(normalisation_target, A),
                                    normalisation_source)
        return A_normalised
