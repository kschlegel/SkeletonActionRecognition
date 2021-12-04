import pytest
import torch

from shar.graphs.graph import Graph
from shar.graphs.graphlayouts import COCO18
from shar.graphs.graphoptions import GraphOptions

coco_distances = torch.tensor(
    [[0, 1, 2, 3, 4, 2, 3, 4, 3, 4, 5, 3, 4, 5, 1, 1, 2, 2],
     [1, 0, 1, 2, 3, 1, 2, 3, 2, 3, 4, 2, 3, 4, 2, 2, 3, 3],
     [2, 1, 0, 1, 2, 2, 3, 4, 1, 2, 3, 3, 4, 5, 3, 3, 4, 4],
     [3, 2, 1, 0, 1, 3, 4, 5, 2, 3, 4, 4, 5, 6, 4, 4, 5, 5],
     [4, 3, 2, 1, 0, 4, 5, 6, 3, 4, 5, 5, 6, 7, 5, 5, 6, 6],
     [2, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 1, 2, 3, 3, 3, 4, 4],
     [3, 2, 3, 4, 5, 1, 0, 1, 4, 5, 6, 2, 3, 4, 4, 4, 5, 5],
     [4, 3, 4, 5, 6, 2, 1, 0, 5, 6, 7, 3, 4, 5, 5, 5, 6, 6],
     [3, 2, 1, 2, 3, 3, 4, 5, 0, 1, 2, 4, 5, 6, 4, 4, 5, 5],
     [4, 3, 2, 3, 4, 4, 5, 6, 1, 0, 1, 5, 6, 7, 5, 5, 6, 6],
     [5, 4, 3, 4, 5, 5, 6, 7, 2, 1, 0, 6, 7, 8, 6, 6, 7, 7],
     [3, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 0, 1, 2, 4, 4, 5, 5],
     [4, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 1, 0, 1, 5, 5, 6, 6],
     [5, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 2, 1, 0, 6, 6, 7, 7],
     [1, 2, 3, 4, 5, 3, 4, 5, 4, 5, 6, 4, 5, 6, 0, 2, 1, 3],
     [1, 2, 3, 4, 5, 3, 4, 5, 4, 5, 6, 4, 5, 6, 2, 0, 3, 1],
     [2, 3, 4, 5, 6, 4, 5, 6, 5, 6, 7, 5, 6, 7, 1, 3, 0, 4],
     [2, 3, 4, 5, 6, 4, 5, 6, 5, 6, 7, 5, 6, 7, 3, 1, 4, 0]])

# TODO: - test adjacency matrices
#       - test directed graphs


class TestGraph:
    @pytest.fixture
    def graph(self, distance, partition_strategy):
        print("Graph:", distance, partition_strategy)
        graph_options = GraphOptions(max_neighbour_distance=distance,
                                     partition_strategy=partition_strategy)
        return Graph(graph_layout=COCO18, graph_options=graph_options)

    @pytest.mark.parametrize("distance,partition_strategy", [(1, "distance"),
                                                             (2, "distance")])
    def test_adjacency_matrix(self, distance, partition_strategy, graph):
        print("Shape:", graph.A.shape)
        assert len(graph.A.shape) == 3

        assert torch.allclose(graph.A[0], torch.eye(graph.A.shape[1]))

        format_str = ",".join(["{:.1f}"] * graph.A.shape[-1])
        for i in range(graph.A.shape[0]):
            print("---> partition", i)
            for j in range(graph.A.shape[1]):
                print(format_str.format(*graph.A[i, j]))

        for d in range(distance):
            assert torch.all(graph.A[d][coco_distances == d] != 0)
            assert torch.all(graph.A[d][coco_distances != d] == 0)

    def test_distance_matrix(self):
        d = Graph._compute_distance_matrix(COCO18.edges, directed_graph=False)
        for i in range(d.shape[0]):
            for j in range(i, d.shape[1]):
                assert d[i, j] == coco_distances[i][j]
                assert d[j, i] == coco_distances[j][i]

    def test_normalisations(self):
        # In the convolution we left-multiply by the feature vector, i.e.
        # matmul(x,A). This leads means row i, col j describes an edge from i
        # to j, so each col below describes the neigbourhood of a vertex
        A = torch.tensor(
            [
                [0, 1, 0, 1, 1],  # edges out of vertex 0
                [0, 0, 0, 0, 0],  # edges out of vertex 1
                [1, 0, 0, 1, 0],  # edges out of vertex 2
                [0, 1, 0, 0, 1],  # edges out of vertex 3
                [0, 1, 0, 0, 0]  # edges out of vertex 4
            ],
            dtype=torch.float)
        # row sums: [3,0,2,2,1]
        # col sums: [1,3,0,2,2]

        # mean pooling normalises the contribution of an edge from node i to
        # node j by the number of neighbours of i. Thus we expect each col
        # (representing the connections from other nodes to i) to be
        # normalised by the sum of col i
        A_mp = Graph._mean_pooling_normalization(A)
        assert torch.allclose(
            A_mp,
            torch.tensor([
                [0, 1 / 3, 0, 1 / 2, 1 / 2],  #
                [0, 0, 0, 0, 0],  #
                [1, 0, 0, 1 / 2, 0],  #
                [0, 1 / 3, 0, 0, 1 / 2],  #
                [0, 1 / 3, 0, 0, 0]  #
            ]))

        # symmetric normalisation normalises the contribution of an edge from
        # node i to node j by the sqrt of the number of edges going out of i
        # and the sqrt of the number of edges coming into j. Thus we expect the
        # entry in row i and col j to be normalised by the sqrt of the sum of
        # row i and the sqrt of the sum of col j
        A_s = Graph._symmetric_normalization(A, directed_graph=True)
        assert torch.allclose(
            A_s,
            torch.sqrt(
                torch.tensor([
                    [0.0, 1 / (3 * 3), 0.0, 1 / (3 * 2), 1 / (3 * 2)],  #
                    [0.0, 0.0, 0.0, 0.0, 0.0],  #
                    [1 / (2 * 1), 0.0, 0.0, 1 / (2 * 2), 0.0],  #
                    [0.0, 1 / (2 * 3), 0.0, 0.0, 1 / (2 * 2)],  #
                    [0.0, 1 / (1 * 3), 0.0, 0.0, 0.0]  #
                ])))
