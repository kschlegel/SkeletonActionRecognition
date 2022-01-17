from __future__ import annotations
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass, asdict

EdgeList = List[Tuple[int, int]]


@dataclass
class GraphLayout:
    """
    Captures the layout of the graph dependent on the data.

    This is primarily the  connections between nodes, whether the graph is
    directed or not on the other hand is not a property of the data but the
    method and can be found in GraphOptions.

    Parameters
    ----------
    edges: List of integer tuples
        List of pairs of integers, describing the connections within the
        graph, excluding self-connections, which are added automatically.
    center_node : int, optional (default is 0)
        The central node of the graph. Only needed when using spatial
        partitioning.
    """
    edges: EdgeList
    center_node: int = 0

    @classmethod
    def fully_connected(cls,
                        num_nodes: int,
                        center_node: int = 0) -> GraphLayout:
        """
        Generate the layout for a fully connected graph.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph
        center_node : int, optional (defaul is 0)
            Center node of the graph in case spatial partitioning is used

        Returns
        -------
        A GraphLayout object for a fully connected graph
        """
        edges = list(
            (i, j) for i in range(num_nodes) for j in range(num_nodes))
        return cls(edges=edges, center_node=center_node)

    @classmethod
    def not_connected(cls, num_nodes: int) -> GraphLayout:
        """
        Generate the layout for a graph without any edges.

        This can be used as a workaround for deactivating the static part of a
        graphs adjacency matrix.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph

        Returns
        -------
        A GraphLayout object for a graph with no edges
        """
        return cls(edges=[], center_node=0)

    def asdict(self) -> Dict[str, Union[EdgeList, int]]:
        """
        Convert to a dictionary.

        Used internally in the graph implementation to be able to convenioently
        unpack the options when needed
        """
        return asdict(self)


COCO18 = GraphLayout(edges=[(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                            (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                            (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)],
                     center_node=1)

KinectV2 = GraphLayout(edges=[(0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
                              (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9),
                              (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
                              (16, 0), (17, 16), (18, 17), (19, 18), (21, 22),
                              (22, 7), (23, 24), (24, 11)],
                       center_node=20)

BerkeleyMHAD = GraphLayout(edges=[(1, 0), (2, 1), (3, 2), (4, 3), (5, 4),
                                  (4, 6), (6, 7), (7, 8), (4, 9), (9, 10),
                                  (10, 11), (0, 12), (12, 13), (13, 14),
                                  (14, 15), (0, 16), (16, 17), (17, 18),
                                  (18, 19)],
                           center_node=4)

Body25 = GraphLayout(edges=[(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6),
                            (6, 7), (1, 8), (8, 9), (9, 10), (10, 11),
                            (11, 22), (22, 23), (11, 24), (8, 12), (12, 13),
                            (13, 14), (14, 19), (19, 20), (14, 21), (0, 15),
                            (15, 17), (0, 16), (16, 18)],
                     center_node=1)

JHMDB = GraphLayout(edges=[(0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (4, 6),
                           (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 12),
                           (9, 13), (10, 14)],
                    center_node=0)


def get_layout_by_datasetname(dataset_name: str) -> GraphLayout:
    """
    Select graph layout based on DatasetLoader class name.

    Provides an easy selector of graph layout object based on the dataset name
    provided for the DatasetLoader module. Raises an exception if the dataset
    name is not one of those supported.

    Parameters
    ----------
    dataset_name : str
        Name of the datset used as passed in as -ds/--dataset command line arg

    Returns
    -------
    graph_layout : GraphLayout object
        The appropriate pre-defined GraphLayout object
    """
    if (dataset_name == "NTURGBD" or dataset_name == "ChaLearn2013"):
        return KinectV2
    elif dataset_name == "Skeletics152":
        return Body25
    elif dataset_name == "BerkeleyMHAD":
        return BerkeleyMHAD
    elif dataset_name == "JHMDB":
        return JHMDB
    else:
        raise Exception("Unknown dataset name.")
