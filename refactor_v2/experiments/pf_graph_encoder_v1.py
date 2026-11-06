"""Small PyTorch graph-attention encoder for the separately prepared SLAKE graph.

This module is not attached to any released model. Message adjacency controls
change graph structure while retaining labels, features and parameter shapes.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GraphAttentionEncoder(nn.Module):
    def __init__(self,d_model=768,heads=8):
        super().__init__();assert d_model%heads==0
        self.heads=heads;self.width=d_model//heads
        self.projection=nn.Linear(d_model,d_model,bias=False)
        self.attention_source=nn.Parameter(torch.empty(heads,self.width))
        self.attention_neighbor=nn.Parameter(torch.empty(heads,self.width))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_source);nn.init.xavier_uniform_(self.attention_neighbor)

    def forward(self,node_features,adjacency):
        assert node_features.ndim==2 and adjacency.ndim==2
        n,d=node_features.shape;assert adjacency.shape==(n,n) and adjacency.dtype==torch.bool
        assert torch.diagonal(adjacency).all(),'Self messages are required for isolated nodes'
        projected=self.projection(node_features).reshape(n,self.heads,self.width)
        source=(projected*self.attention_source).sum(-1)
        neighbor=(projected*self.attention_neighbor).sum(-1)
        logits=F.leaky_relu(source[:,None,:]+neighbor[None,:,:],negative_slope=.2).permute(2,0,1)
        logits=logits.masked_fill(~adjacency.unsqueeze(0),float('-inf'))
        weights=logits.softmax(-1)
        values=projected.permute(1,0,2)
        output=torch.matmul(weights,values).permute(1,0,2).reshape(n,d)
        return F.elu(output)


def adjacency_from_edges(n,edges,device='cpu'):
    adjacency=torch.zeros(n,n,dtype=torch.bool,device=device)
    pairs=torch.as_tensor(edges,dtype=torch.long,device=device)
    assert pairs.ndim==2 and pairs.shape[1]==2 and (pairs>=0).all() and (pairs<n).all()
    adjacency[pairs[:,0],pairs[:,1]]=True
    assert torch.diagonal(adjacency).all()
    return adjacency
