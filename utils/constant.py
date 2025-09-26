import torch

UNITY_TO_OPENCV=(1,-1,1,-1,1,-1)

T_NED_TO_RDF = torch.tensor([[0,1,0,0],
                             [0,0,1,0],
                             [1,0,0,0],
                             [0,0,0,1]])