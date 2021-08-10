"""
This implementation is based on
https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py
"""
from typing import List, Union, Tuple, Optional

import torch


class Graph(torch.nn.Module):
    def __init__(self,
                 graph_layout: Optional[str] = 'kinectv2',
                 graph_connections: Union[List[Tuple[int]], None] = None,
                 center_node: Optional[int] = None,
                 graph_partition_strategy: str = 'uniform',
                 directed_graph: bool = False,
                 learnable_adjacency: bool = False,
                 data_dependent_adjacency: bool = False,
                 edge_importance_weighting: bool = False,
                 normalisation_method: str = "mean_pooling",
                 max_neighbour_distance: int = 1,
                 dilation: int = 1,
                 in_channels: Optional[int] = None,
                 embedding_dimension: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        graph_layout : str, Optional (default is 'kinectv2)
            One of ('COCO18', 'kinectv2') - String identifier of the graph
            layout for some common keypoint collections.
            If not provided then both connections and center_node must be given
        graph_connections: List of integer tuples, Optional (default is None)
            List of pairs of integers, describing the connections within the
            graph, excluding self-connections, which are added automatically.
            When using this centre_node must also be specified.
            If not provided then partition_strategy must be given
        center_node: int, Optional (default is None)
            ID of the centre node of the skeleton for spatial partition
            strategy. Must be provided in conjunction with connections list.
            If not provided then partition_strategy must be given
        graph_partition_strategy : str, Optional (default is 'uniform)
            One of ('uniform','distance','spatial') - Graph partition strategy
            defining the weight function
        directed_graph : bool, Optional (default is False)
            Whether to use a directed or undirected graph. If set to True, the
            pairs of integers in graph_connections are interpreted as the first
            integer denoting the start of an edge and the second integer its
            end.
        learnable_adjacency : bool, optional (default is False)
            Whether to include a learnable component in the adjacency matrix
        data_dependent_adjacency : bool, optinal (default is False)
            Whether to include a data dependent component in the adjacency
            matrix
        edge_importance_weighting : bool, optional (default is False)
            Whether to include a learnable importance weighting to the edges of
            the graph
        normalisation_method : str, optional (default is 'mean_pooling')
            One of ('mean_pooling', 'symmetric') - String identifying the
            normalisation method applied to the adjacency matrix to normalise
            contributions based on group sizes
        max_neighbour_distance : int, Optional (default is 1)
            Maximal distance between to neighbours to be considered connected
            for forming a nodes neighbour set.
        dilation : int, Optional (default is 1)
            Step size of distance for connections to be considered for the
            neighbour of a node. For example, if the maximal distance is set to
            4 with a dilation of 2 then only nodes with distance 0,2 and 4 are
            considered part of the neighbour set of a node.
        in_channels : int
            Dimension of the data at each vertex.
            Only needs to be supplied when using a data-dependent component in
            the adjacency matrix.
        embedding_dimension : int
            Dimension of the data embedding for the computation of the
            data-dependent component of the adjacency matrix.
            Only needs to be supplied when using a data-dependent component in
            the adjacency matrix.
        """
        super().__init__()

        self.directed_graph = directed_graph
        self._compute_edges(graph_layout=graph_layout,
                            connections=graph_connections,
                            center_node=center_node)
        self._compute_distance_matrix(max_neighbour_distance)

        # Static component of the adjacency matrix
        self.A: torch.Tensor
        self.register_buffer(
            "A",
            self._get_adjacency_matrix(
                partition_strategy=graph_partition_strategy,
                normalisation_method=normalisation_method,
                max_neighbour_distance=max_neighbour_distance,
                dilation=dilation))

        # Optional learnable component for the adjacency matrix
        self.learnable_adjacency = learnable_adjacency
        if self.learnable_adjacency:
            self.B = torch.nn.Parameter(torch.zeros(self.A.shape))

        # Optional data-dependent component for the adjacency matrix
        self.data_dependent_adjacency = data_dependent_adjacency
        if self.data_dependent_adjacency:
            if in_channels is None or embedding_dimension is None:
                raise Exception("The input and embedding dimensions needs to "
                                "be specified when using a data-dependent "
                                "component in the adjacency matrix.")
            self.phi = torch.nn.ModuleList()
            self.theta = torch.nn.ModuleList()
            self.embedding_dimension = embedding_dimension
            for i in range(self.A.shape[0]):
                self.phi.append(
                    torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=embedding_dimension,
                                    kernel_size=1))
                self.theta.append(
                    torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=embedding_dimension,
                                    kernel_size=1))

        # Optional learnable importance weighting for each edge in the graph
        self.edge_importance: Union[torch.Tensor, None]
        if edge_importance_weighting:
            print("Add edge importance")
            self.edge_importance = torch.nn.Parameter(torch.ones(
                self.A.size()))
        else:
            self.edge_importance = None

    def forward(self, x):
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
            C = torch.zeros((x.shape[0], ) + self.A.shape)
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

    def _compute_edges(self,
                       graph_layout: Optional[str] = None,
                       connections: Union[List[Tuple[int]], None] = None,
                       center_node: Optional[int] = None) -> None:
        """
        Computes the list of edges in the graph.

        When passed a string identifier of a graph layout, creates a list of
        all edges (expressed as a tuple of node indices), including self-links
        of all nodes and stores it in self.edges.
        self.num_nodes contains the number of nodes in the graph, and
        self.centre represents the shoulder centre/neck keypoint for the
        spatial graph partitioning strategy.

        Can also be passed a list of connections in the graph, together with
        the id of centre node. In this case the number of nodes is inferred
        from the indices occuring in the connection list and self connections
        of all nodes are added to the list.

        Parameters
        ----------
        graph_layout : str, Optional (default is None)
            One of ('COCO18', 'kinectv2') - String identifier of the graph
            layout for some common keypoint collections.
            If not provided then both connections and center_node must be given
        connections: List of integer tuples, Optional (default is None)
            List of pairs of integers, describing the connections within the
            graph, excluding self-connections, which are added automatically.
            When using this centre_node must also be specified.
            If not provided then partition_strategy must be given
        center_node: int, Optional (default is None)
            ID of the centre node of the skeleton for spatial partition
            strategy. Must be provided in conjunction with connections list.
            If not provided then partition_strategy must be given
        """
        if connections is not None and center_node is not None:
            self.num_nodes = max(max(c) for c in connections) + 1
            self.center = center_node
        elif graph_layout is not None:
            if graph_layout == 'COCO18':
                self.num_nodes = 18
                connections = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12),
                               (12, 11), (10, 9), (9, 8), (11, 5), (8, 2),
                               (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                               (17, 15), (16, 14)]
                self.center = 1
            elif graph_layout == 'kinectv2':
                self.num_nodes = 25
                connections = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
                               (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                               (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
                               (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
                               (21, 22), (22, 7), (23, 24), (24, 11)]
                self.center = 20
            else:
                raise ValueError("Invalid layout.")
        else:
            raise Exception("Either graph_layout or connections&centre need to"
                            " be provided to create the graph.")
        self_connections = [(i, i) for i in range(self.num_nodes)]
        self.edges = self_connections + connections

    def _get_adjacency_matrix(self, partition_strategy: str,
                              normalisation_method: str,
                              max_neighbour_distance: int,
                              dilation: int) -> torch.Tensor:
        """
        Computes the adjacency matrix A for different partitioning strategies.

        When a partition strategy with more than one subgroup is used (i.e.
        anything != uniform), then the adjacency matrix is split into several
        individual matrices for which the output is computed and then summed
        over all matrix instances, i.e. A=sum_j A_j and
        x_out = sum_j A_j*x_in*w
        The returned adjacency matrices are already nornmalised by subset
        cardinalities using the chosen normalisation method.

        TODO: Spatial partitioning is not working as expected:
                The paper describes the subgrouping as based on distance to the
                root, but here the groups are formed by comparing both nodes
                distance to root using the distance matrix, which has inf
                values wherever we are above max_neighbour_distance. This means
                e.g. for max_distance=1, as the right elbow has distance 2 to
                the neck the distance matrix gives it distance inf and _every_
                neighbour of the elbow has a distance less or equal to it so
                ends up in the centripetal group.

        Parameters
        ----------
        partition_strategy : str
            One of ('uniform','distance','spatial') - Identifier of the
            parition strategy to use.
            Uniform partition: All nodes in a neighbourhood get assigned the
            same label
            Distance partition: The root node forms one subgroup, all other
            nodes are grouped in another subgroup
            Spatial partition: Three subgroups: Nodes closer to the skeletons
            centre than the root node (centripetal nodes), nodes of the same
            distance to the skeletons centre, and nodes further away from the
            skeletons centre (centrifugal nodes)
        normalisation_method : str
            One of ('mean_pooling', 'symmetric') - String identifying the
            normalisation method applied to the adjacency matrix to normalise
            contributions based on group sizes
        max_neighbour_distance : int
            Maximal distance between to neighbours to be considered connected
            for forming a nodes neighbour set.
        dilation : int
            Step size of distance for connections to be considered for the
            neighbour of a node. For example, if the maximal distance is set to
            4 with a dilation of 2 then only nodes with distance 0,2 and 4 are
            considered part of the neighbour set of a node.

        Returns
        -------
        A : tensor
            Set of normalised adjacency matrices (normalised by subset
            cardinalities)
        """
        valid_steps = range(0, max_neighbour_distance + 1, dilation)
        adjacency = torch.zeros((self.num_nodes, self.num_nodes))
        for step in valid_steps:
            adjacency[self.distance_matrix == step] = 1
        if normalisation_method == "mean_pooling":
            normalized_adjacency = self._mean_pooling_normalization(adjacency)
        elif normalisation_method == "symmetric":
            normalized_adjacency = self._symmetric_normalization(adjacency)
        else:
            raise Exception("Invalid normalisation method.")

        if partition_strategy == 'uniform':
            # Uniform partitioning leads to a single adjacency matrix
            # containing all connections
            A = torch.unsqueeze(normalized_adjacency, dim=0)
        elif partition_strategy == 'distance':
            # For distance partitioning the adjacency matrix splits directly
            # according to the distance, with each A_j containing the
            # connections at distance j
            A = torch.zeros((len(valid_steps), self.num_nodes, self.num_nodes))
            for i, step in enumerate(valid_steps):
                A[i][self.distance_matrix == step] = normalized_adjacency[
                    self.distance_matrix == step]
        elif partition_strategy == 'spatial':
            A_list: List[torch.Tensor] = []
            for step in valid_steps:
                # root: same distance to the centre node as the root
                A_root = torch.zeros((self.num_nodes, self.num_nodes))
                # Centripetal: closer to the center node than the root
                A_centripetal = torch.zeros((self.num_nodes, self.num_nodes))
                # Centrifugal: further from the center node than the root
                A_centrifugal = torch.zeros((self.num_nodes, self.num_nodes))
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if self.distance_matrix[j, i] == step:
                            if (self.distance_matrix[j, self.center] ==
                                    self.distance_matrix[i, self.center]):
                                A_root[j, i] = normalized_adjacency[j, i]
                            elif (self.distance_matrix[j, self.center] >
                                  self.distance_matrix[i, self.center]):
                                A_centripetal[j, i] = normalized_adjacency[j,
                                                                           i]
                            else:
                                A_centrifugal[j, i] = normalized_adjacency[j,
                                                                           i]
                if step == 0:
                    A_list.append(A_root)
                else:
                    A_list.append(A_root + A_centripetal)
                    A_list.append(A_centrifugal)
            A = torch.stack(A_list, dim=0)
        else:
            raise ValueError("Invalid partition strategy")
        A.requires_grad = False
        return A

    def _compute_distance_matrix(self, max_neighbour_distance: int) -> None:
        """
        Returns a matrix of pairwise node distances.

        Returns a matrix of shape (num_nodes, num_nodes) where each entry
        contains the distance between nodes i and j. Pairs further apart than
        the maximal distance have their distance set to infinity.

        Parameters
        ----------
        max_neighbour_distance : int
            Maximal distance between to neighbours to be considered connected
            for forming a nodes neighbour set.
        """
        A = torch.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.edges:
            A[i, j] = 1  # edge from i to j
            if not self.directed_graph:
                A[j, i] = 1  # edge from j to i

        self.distance_matrix = torch.full((self.num_nodes, self.num_nodes),
                                          float("Inf"))
        # Powers A^i for i=0,...,max_neighbour_distance of the adjacency matrix
        # will each be non-zero at each pair that has at most distance i.
        matrix_powers = [
            torch.matrix_power(A, d) for d in range(max_neighbour_distance + 1)
        ]
        # Putting distances into the distance matrix in decending order
        # iteratively overrides the smaller distances until the correct
        # distance
        for d in range(max_neighbour_distance, -1, -1):
            self.distance_matrix[matrix_powers[d] > 0] = d

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

    def _symmetric_normalization(self, A: torch.Tensor) -> torch.Tensor:
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
        if self.directed_graph:
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
