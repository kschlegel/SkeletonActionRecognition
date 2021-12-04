from typing import Union
from dataclasses import dataclass, asdict
from functools import partial

from .graphpartitions import GraphPartition


@dataclass
class GraphOptions:
    """
    Captures the properties of the graph used in a given model.

    Parameters
    ----------
    directed_graph : bool, Optional (default is False)
        Whether to use a directed or undirected graph. If set to True, the
        pairs of integers in graph_connections are interpreted as the first
        integer denoting the start of an edge and the second integer its
        end.
    max_neighbour_distance : int, Optional (default is 1)
        Maximal distance between to neighbours to be considered connected
        for forming a nodes neighbour set.
    dilation : int, Optional (default is 1)
        Step size of distance for connections to be considered for the
        neighbour of a node. For example, if the maximal distance is set to
        4 with a dilation of 2 then only nodes with distance 0,2 and 4 are
        considered part of the neighbour set of a node.
    partition_strategy : GraphPartition obj or str, Optional (default
        is 'uniform)
        Graph partition strategy defining the weight function.
        If str, one of ('uniform','distance','spatial'):
        - Uniform partition: All nodes in a neighbourhood get assigned the
        same label
        - Distance partition: The root node forms one subgroup, all other
        nodes are grouped in another subgroup
        - Spatial partition: Three subgroups: Nodes closer to the skeletons
        centre than the root node (centripetal nodes), nodes of the same
        distance to the skeletons centre, and nodes further away from the
        skeletons centre (centrifugal nodes)

        Can also be an object of a class inheriting from GraphPartition
        base class. See GraphPartition documentation for more details.
    include_self_loops : bool, optional (default is False)
        Only relevant for distance partitioning. If True self loops are
        included in the adjacency component for each individual distance
        and not just in the distance 0 component
    normalisation_method : str, optional (default is 'mean_pooling')
        One of ('mean_pooling', 'symmetric') - String identifying the
        normalisation method applied to the adjacency matrix to normalise
        contributions based on group sizes
    normalise_first: bool, Optional (default is False)
        Whether to normalise the adjacency matrix before or after
        partitioning, i.e. based on overall neighbour counts or partition
        neighbour counts.
    edge_importance_weighting : bool, optional (default is False)
        Whether to include a learnable importance weighting to the edges of
        the graph
    learnable_adjacency : bool, optional (default is False)
        Whether to include a learnable component in the adjacency matrix
    data_dependent_adjacency : bool, optional (default is False)
        Whether to include a data dependent component in the adjacency
        matrix
    """
    directed_graph: bool = False
    max_neighbour_distance: int = 1
    dilation: int = 1
    partition_strategy: Union[str, GraphPartition] = 'uniform'
    include_self_loops: bool = False
    normalisation_method: str = "mean_pooling"
    normalise_first: bool = False
    edge_importance_weighting: bool = False
    learnable_adjacency: bool = False
    data_dependent_adjacency: bool = False

    def asdict(self):
        """
        Convert to a dictionary.

        Used internally in the graph implementation to be able to convenioently
        unpack the options when needed
        """
        return asdict(self)


# partition strategy and edge_importance_weighting are the parameters varied in
# the paper
ST_GCN_Options = partial(GraphOptions,
                         max_neighbour_distance=1,
                         dilation=1,
                         include_self_loops="False",
                         normalisation_method="mean_pooling",
                         normalise_first=True,
                         learnable_adjacency=False,
                         data_dependent_adjacency=False)

# learnable_adjacency and data_dependent_adjacency are the parameters specific
# to this paper
AGCN_Options = partial(GraphOptions,
                       max_neighbour_distance=1,
                       dilation=1,
                       partition_strategy='uniform',
                       include_self_loops=False,
                       normalisation_method="mean_pooling",
                       normalise_first=True,
                       edge_importance_weighting=False)

# max_neighbour_distance is the main parameter for this module in the paper
MS_GCN_Options = partial(GraphOptions,
                         dilation=1,
                         partition_strategy="distance",
                         include_self_loops=False,
                         normalisation_method="symmetric",
                         normalise_first=False,
                         edge_importance_weighting=False,
                         learnable_adjacency=True,
                         data_dependent_adjacency=False)
