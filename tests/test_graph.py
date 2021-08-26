import pytest
import torch

from shar.graphs import Graph

coco_distances = [[0, 1, 2, 3, 4, 2, 3, 4, 3, 4, 5, 3, 4, 5, 1, 1, 2, 2],
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
                  [2, 3, 4, 5, 6, 4, 5, 6, 5, 6, 7, 5, 6, 7, 3, 1, 4, 0]]

# TODO: - test adjacency matrices
#       - test directed graphs


class TestGraph:
    @pytest.fixture
    def graph(self, distance, partition_strategy="uniform"):
        return Graph(graph_layout="COCO18",
                     graph_partition_strategy=partition_strategy,
                     max_neighbour_distance=distance)

    @pytest.mark.parametrize("distance", [1, 2])
    def test_distance_matrix(self, distance, graph):
        d = graph.distance_matrix
        for i in range(d.shape[0]):
            for j in range(i, d.shape[1]):
                if coco_distances[i][j] <= distance:
                    assert d[i, j] == coco_distances[i][j]
                    assert d[j, i] == coco_distances[j][i]
                else:
                    assert d[i, j] == float("Inf")
                    assert d[j, i] == float("Inf")

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
        g = Graph(directed_graph=True)
        A_s = g._symmetric_normalization(A)
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
