from typing import List
from abc import ABC, abstractmethod

import torch


class GraphPartition(ABC):
    """
    Strategy to partition neighbour sets of the graph.

    This is the base class for classes implementing a strategy to partition the
    neighbour sets of nodes of a graph. The constructor can be used to define
    certain global properties such as e.g. a central node to relate others to.

    The partitioning is done in the apply method which receives a (possibly
    normalised) adjacency matrix, the distance matrix encoding all pairwise
    distances between nodes, and a list of individual distance steps used to
    form neighbourhood sets.
    The adjacency_matrix is nonzero at i,j if node j is a neighbour of i for
    any of the given distance steps (the value will be 1 if the adjacency
    matrix has not yet been normalised, <= 1 otw. Thus the partitioning should
    happen by copying these values over into their partition to maintain
    normalisation before partitioning)
    """
    @abstractmethod
    def apply(self, adjacency_matrix: torch.Tensor,
              distance_matrix: torch.Tensor,
              distance_steps: List[int]) -> torch.Tensor:
        ...


class UniformPartition(GraphPartition):
    """
    All nodes in a neighbourhood get assigned the same label
    """
    def apply(self, adjacency_matrix: torch.Tensor,
              distance_matrix: torch.Tensor,
              distance_steps: List[int]) -> torch.Tensor:
        return torch.unsqueeze(adjacency_matrix, dim=0)


class DistancePartition(GraphPartition):
    """
    All nodes at the same distance to the root node form one group.

    I.e. the root node forms one subgroup and for each distance d in the
    distance_steps list all nodes at that distance from the root node form
    another subgroup.
    """
    def __init__(self, include_self_loops: bool = False, **kwargs):
        self._include_self_loops = include_self_loops

    def apply(self, adjacency_matrix: torch.Tensor,
              distance_matrix: torch.Tensor,
              distance_steps: List[int]) -> torch.Tensor:
        A = torch.zeros((len(distance_steps), ) + adjacency_matrix.shape)
        for i, d in enumerate(distance_steps):
            A[i][distance_matrix == d] = adjacency_matrix[distance_matrix == d]
            if self._include_self_loops and d > 0:
                A[i] += torch.eye(A[i].shape[0], dtype=A[i].dtype)
        return A


class SpatialPartition(GraphPartition):
    """
    Partitions by distance from the root node and the graphs center node.

    First groups by distance just as distance partitioning. Then for each
    distance d>0 splits up the distance partition into two groups, nodes closer
    or as close as the root node to the skeletons center node (centripetal
    nodes) and nodes further away from the skeletons centre (centrifugal nodes)
    """
    def __init__(self, center_node: int, **kwargs):
        self._center_node = center_node

    def apply(self, adjacency_matrix: torch.Tensor,
              distance_matrix: torch.Tensor,
              distance_steps: List[int]) -> torch.Tensor:
        A_list: List[torch.Tensor] = []
        for d in distance_steps:
            # root: same distance to the centre node as the root
            A_root = torch.zeros(adjacency_matrix.shape)
            # Centripetal: closer to the center node than the root
            A_centripetal = torch.zeros(adjacency_matrix.shape)
            # Centrifugal: further from the center node than the root
            A_centrifugal = torch.zeros(adjacency_matrix.shape)
            for i in range(adjacency_matrix.shape[1]):
                for j in range(adjacency_matrix.shape[0]):
                    if distance_matrix[j, i] == d:
                        if (distance_matrix[j, self._center_node] ==
                                distance_matrix[i, self._center_node]):
                            A_root[j, i] = adjacency_matrix[j, i]
                        elif (distance_matrix[j, self._center_node] >
                              distance_matrix[i, self._center_node]):
                            A_centripetal[j, i] = adjacency_matrix[j, i]
                        else:
                            A_centrifugal[j, i] = adjacency_matrix[j, i]
            if d == 0:
                A_list.append(A_root)
            else:
                A_list.append(A_root + A_centripetal)
                A_list.append(A_centrifugal)
        return torch.stack(A_list, dim=0)
