import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Tuple

# Assume that your VarCLR pretrained encoder is already available
from varclr.models.model import Encoder

# Initialize VarCLR's pretrained model (used for textual embeddings)
model = Encoder.from_pretrained("varclr-codebert")
torch.manual_seed(42)  # Set the seed for reproducibility

# Set up CodeBERT for textual embeddings (as a proxy)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
text_model = AutoModel.from_pretrained("microsoft/codebert-base")
text_model.eval()


def get_textual_embedding(text: str) -> torch.Tensor:
    """
    Compute a 768-dimensional embedding for a given variable name.
    """
    vector = model.encode(text)  # shape: (1, 768)
    return vector.squeeze(0)  # shape: (768,)


# ---------------------------
# Define DGI: Deep Graph Infomax Framework
# ---------------------------
class DGIEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """
        A simple 2-layer GCN encoder that maps input features to latent node embeddings.

        Args:
            in_channels (int): Dimension of input features.
            hidden_channels (int): Hidden dimension of the first GCN layer.
            out_channels (int): Output dimension (final node embedding size).
        """
        super(DGIEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First layer with non-linearity.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second layer produces final embeddings.
        x = self.conv2(x, edge_index)
        return x  # shape: (num_nodes, out_channels)


class DGI(nn.Module):
    def __init__(self, encoder: DGIEncoder, summary_func=torch.sigmoid):
        """
        Deep Graph Infomax (DGI) module.

        Args:
            encoder (DGIEncoder): An encoder module that computes node embeddings.
            summary_func (callable): Function applied on the global summary vector (default: sigmoid).
        """
        super(DGI, self).__init__()
        self.encoder = encoder
        self.summary_func = summary_func
        # Discriminator: a bilinear layer to compare node embeddings with the summary vector.
        self.disc = nn.Bilinear(self.encoder.conv2.out_channels, self.encoder.conv2.out_channels, 1)

    def readout(self, H: torch.Tensor) -> torch.Tensor:
        """
        Compute a global summary vector by averaging node embeddings and applying a non-linearity.
        """
        s = torch.mean(H, dim=0)  # Global average pooling over nodes.
        s = self.summary_func(s)
        return s  # shape: (out_channels,)

    def forward(self, x, edge_index):
        """
        Forward pass that computes discriminator scores for real and corrupted embeddings.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Graph connectivity.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Scores for positive (real) and negative (corrupted) examples.
        """
        # Compute real node embeddings.
        H = self.encoder(x, edge_index)  # (num_nodes, out_channels)
        s = self.readout(H)  # (out_channels,)
        s_expanded = s.unsqueeze(0).expand_as(H)  # (num_nodes, out_channels)
        pos = self.disc(H, s_expanded)  # Scores for true embeddings.

        # Create corrupted embeddings by shuffling input features.
        perm = torch.randperm(x.size(0))
        x_corrupted = x[perm]
        H_corrupted = self.encoder(x_corrupted, edge_index)
        neg = self.disc(H_corrupted, s_expanded)  # Scores for corrupted embeddings.

        return pos, neg

    def loss(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss for DGI.

        Args:
            pos (torch.Tensor): Discriminator scores for real embeddings.
            neg (torch.Tensor): Discriminator scores for corrupted embeddings.

        Returns:
            torch.Tensor: Scalar loss.
        """
        lbl_pos = torch.ones_like(pos)
        lbl_neg = torch.zeros_like(neg)
        pos_loss = F.binary_cross_entropy_with_logits(pos, lbl_pos)
        neg_loss = F.binary_cross_entropy_with_logits(neg, lbl_neg)
        return pos_loss + neg_loss


def encode_variable(variable: str, node_order: List[str], H: torch.Tensor, fallback_proj: nn.Module) -> torch.Tensor:
    """
    Retrieve the embedding for a variable.
    If the variable is in the graph (node_order), return its corresponding DGI embedding.
    Otherwise, compute its textual embedding, fuse it with zeros for the structural part,
    and pass it through fallback_proj so that the fallback embedding has the same dimension
    as the DGI encoder output.

    Args:
        variable (str): Variable name.
        node_order (List[str]): List of node names corresponding to rows in H.
        H (torch.Tensor): Node embeddings from the DGI encoder (shape: num_nodes x embedding_dim).
        fallback_proj (nn.Module): Projection layer to map the fused feature (772) to embedding size.

    Returns:
        torch.Tensor: The embedding for the variable.
    """
    if variable in node_order:
        idx = node_order.index(variable)
        return H[idx]
    else:
        text_emb = get_textual_embedding(variable)  # shape: (768,)
        # Create fused feature: textual embedding concatenated with zeros for structural features.
        fused_feat = torch.cat([text_emb, torch.zeros(4, dtype=torch.float)], dim=0)  # shape: (772,)
        # Project to the DGI output dimension.
        fallback_embedding = fallback_proj(fused_feat.unsqueeze(0)).squeeze(0)
        return fallback_embedding


# ---------------------------
# Build the graph and fused features.
# ---------------------------
graphml_file = "Graphs/global_variable_dependency_graph.graphml"
global_vdg = nx.read_graphml(graphml_file)

# Construct a consistent node order.
node_order = list(global_vdg.nodes())

# Build fused features: textual embedding concatenated with 4 structural features.
fused_features = {}  # Dictionary: node -> fused feature (tensor)
for node in node_order:
    # Retrieve structural features as a comma-separated string; if missing, use "0,0,0,0".
    structural_str = global_vdg.nodes[node].get("x", "0,0,0,0")
    structural_feat = torch.tensor([float(x.strip()) for x in structural_str.split(",") if x.strip() != ""],
                                   dtype=torch.float)
    text_emb = get_textual_embedding(node)  # shape: (768,)
    fused_feat = torch.cat([text_emb, structural_feat], dim=0)  # shape: (772,)
    fused_features[node] = fused_feat
    # Update node attribute for potential GraphML saving.
    global_vdg.nodes[node]["x"] = fused_feat.tolist()

data = from_networkx(global_vdg)
# Ensure data.x has the nodes in the same order as node_order.
data.x = torch.stack([fused_features[node] for node in node_order], dim=0)

# ---------------------------
# Initialize and train the DGI model.
# ---------------------------
in_channels = data.x.size(1)  # Should be 772 (textual embedding + 4 structural features)
hidden_channels = 128
out_channels = 128 # Desired dimension for the DGI encoder's output

dgi_encoder = DGIEncoder(in_channels, hidden_channels, out_channels)
dgi_model = DGI(dgi_encoder)
optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.003)

dgi_model.train()
num_epochs = 200
for epoch in range(num_epochs):
    torch.manual_seed(42)
    optimizer.zero_grad()
    pos, neg = dgi_model(data.x, data.edge_index)
    loss = dgi_model.loss(pos, neg)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: DGI Loss = {loss.item():.4f}")

# ---------------------------
# Define a fallback projection layer.
# This projects a fused feature of size 772 to an embedding of size 128.
# ---------------------------
fallback_proj = nn.Linear(data.x.size(1), out_channels)
fallback_proj.eval()  # No training is done for this layer in this script.

# ---------------------------
# Evaluate on benchmark pairs.
# ---------------------------
dgi_model.eval()
with torch.no_grad():
    # Extract node embeddings from the trained encoder.
    H = dgi_model.encoder(data.x, data.edge_index)

try:
    from varclr.benchmarks import Benchmark

    benchmarks = [
        Benchmark.build("idbench", variant="medium", metric="similarity"),
        Benchmark.build("idbench", variant="medium", metric="relatedness"),
        Benchmark.build("idbench", variant="small", metric="similarity"),
        Benchmark.build("idbench", variant="small", metric="relatedness"),
        Benchmark.build("idbench", variant="large", metric="similarity"),
        Benchmark.build("idbench", variant="large", metric="relatedness")
    ]

    for bench in benchmarks:
        id1_list, id2_list = bench.get_inputs()
        # For each benchmark variable, use the graph embedding if available; otherwise, use fallback.
        emb_id1 = torch.stack([encode_variable(var, node_order, H, fallback_proj) for var in id1_list], dim=0)
        emb_id2 = torch.stack([encode_variable(var, node_order, H, fallback_proj) for var in id2_list], dim=0)
        # Normalize embeddings prior to computing cosine similarity.
        emb_id1 = F.normalize(emb_id1, p=2, dim=1)
        emb_id2 = F.normalize(emb_id2, p=2, dim=1)
        predicted = F.cosine_similarity(emb_id1, emb_id2)
        predicted_list = predicted.tolist()
        print(f"\nBenchmark evaluation ({bench.variant} - {bench.metric}):")
        print(bench.evaluate(predicted_list))
except Exception as e:
    print("Benchmark evaluation skipped or encountered an error:", e)

if __name__ == "__main__":
    print("Training complete.")