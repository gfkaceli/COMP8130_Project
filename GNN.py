# gnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from varclr.models.model import Encoder
from typing import List

# Load VarCLR's pretrained encoder for textual embeddings
_varclr_model = Encoder.from_pretrained("varclr-codebert")

def get_textual_embedding(name: str) -> torch.Tensor:
    """
    Compute a 768-dimensional embedding for a given variable name using VarCLR.
    """
    vec = _varclr_model.encode(name)      # shape: (1, 768)
    return vec.squeeze(0)                 # shape: (768,)

class DGIEncoder(nn.Module):
    """
    A 2-layer GCN that maps fused features to 128-dim node embeddings.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)  # (num_nodes, out_channels)

class DGI(nn.Module):
    """
    Deep Graph Infomax module: maximizes mutual info between node embeddings and a graph summary.
    """
    def __init__(self, encoder: DGIEncoder, summary_fn=torch.sigmoid):
        super().__init__()
        self.encoder = encoder
        self.summary_fn = summary_fn
        d = encoder.conv2.out_channels
        self.discriminator = nn.Bilinear(d, d, 1)

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        # Global summary: mean over nodes + nonlinearity
        s = H.mean(dim=0)
        return self.summary_fn(s)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        H = self.encoder(x, edge_index)        # real node embeddings
        s = self.readout(H)                    # global summary
        s_exp = s.unsqueeze(0).expand_as(H)    # match nodes
        pos_scores = self.discriminator(H, s_exp)

        # Corrupted (negative) examples by shuffling features
        perm = torch.randperm(x.size(0))
        H_corrupt = self.encoder(x[perm], edge_index)
        neg_scores = self.discriminator(H_corrupt, s_exp)

        return pos_scores, neg_scores

    def loss(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        # BCE loss: real → 1, corrupt → 0
        lbl_pos = torch.ones_like(pos)
        lbl_neg = torch.zeros_like(neg)
        return (F.binary_cross_entropy_with_logits(pos, lbl_pos)
              + F.binary_cross_entropy_with_logits(neg, lbl_neg))

def encode_variable(
    name: str,
    node_order: List[str],
    H: torch.Tensor,
    fallback_proj: nn.Module
) -> torch.Tensor:
    """
    Retrieve embedding for 'name'. If it's in node_order, return H[idx].
    Otherwise compute a 772‑dim fused feature (768 from text + 4 zeros)
    and project it to the same 128‑dim space via fallback_proj.
    """
    if name in node_order:
        return H[node_order.index(name)]
    text_emb = get_textual_embedding(name)                  # (768,)
    fused = torch.cat([text_emb, torch.zeros(4)], dim=0)    # (772,)
    return fallback_proj(fused.unsqueeze(0)).squeeze(0)     # (128,)