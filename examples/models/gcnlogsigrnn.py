import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shar.graphs import MultiScale_GraphConv as MS_GCN
from shar.signatures import LogSigRNN
from shar.graphs.ntu_rgb_d import AdjMatrixGraph


class GCNLOGSIG(nn.Module):
    def __init__(self,
                 num_classes,
                 num_point=25,
                 num_person=2,
                 num_gcn_scales=13,
                 graph='shar.graphs.ntu_rgb_d.AdjMatrixGraph',
                 in_channels=3,
                 **kwargs):
        super(GCNLOGSIG, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 96
        c2 = c1 * 2
        self.c1 = c1
        self.c2 = c2

        self.gcn1 = MS_GCN(num_gcn_scales, 3, c1,
                           A_binary, disentangled_agg=True)

        self.n_segments1 = 50
        self.logsigrnn1 = LogSigRNN(
                 in_channels = c1,
                 logsignature_lvl = 2,
                 num_segments = self.n_segments1,
                 out_channels = c1,
            )
        """
        self.gcn2 = MS_GCN(num_gcn_scales, c1, c2,
                           A_binary, disentangled_agg=True)

        self.n_segments2 = 30
        self.logsigrnn2 = LogSigRNN(
                 in_channels = c2,
                 logsignature_lvl = 2,
                 num_segments = self.n_segments2,
                 out_channels = c2,
            )
        """
        self.fc = nn.Linear(c1, num_classes)

    def forward(self, x):
        N, M, C, T, V = x.size()
        
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        # N,C,T,V

        x = F.relu(self.gcn1(x), inplace=False)
        x = self.logsigrnn1(x)

        #x = F.relu(self.gcn2(x), inplace=False)
        #x = self.logsigrnn2(x)

        out = x
        
        out_channels = out.size(1)
        
        out = out.reshape(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod