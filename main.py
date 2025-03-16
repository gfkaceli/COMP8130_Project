import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from tensorboard.compat.tensorflow_stub.dtypes import variant
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from transformers import AutoTokenizer, AutoModel
from varclr.benchmarks import Benchmark
from varclr.models.model import Encoder
from typing import List, Union

# ---------------------------
# Initialize VarCLR's Pretrained Encoder
# ---------------------------
# Load VarCLR's pretrained model (using it as our textual encoder).
model = Encoder.from_pretrained("varclr-codebert")
torch.manual_seed(42) # set the seed
# ---------------------------
# Step 1: Define a small vocabulary of 10 variable names.
# ---------------------------
vocab = ['x', 'number', 'prop', 'store', 'values', 'converted', 'valueswave', 'stream', 'counter', 'result']
print("Vocabulary:", vocab)


# ---------------------------
# Step 2: Set up CodeBERT (as a proxy for VarCLR's textual encoder) to get textual embeddings.
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
text_model = AutoModel.from_pretrained("microsoft/codebert-base")
text_model.eval()  # using CPU

def get_textual_embedding(text: str) -> torch.Tensor:
    """
    Generate a 768-dim textual embedding for a given variable name using VarCLR's encoder.
    """
    vector = model.encode(text)  # shape: (1, 768)
    return vector.squeeze(0)

# ---------------------------
# Step 3: Create fused node features.
# For each variable, generate its textual embedding and append 5 random structural features.
# ---------------------------
node_features = {}
for var in vocab:
    text_emb = get_textual_embedding(var)  # (768,)
    # Generate 5 random structural features (for demo purposes)
    structural_feat = torch.tensor([random.random() for _ in range(5)], dtype=torch.float)
    # Fuse features by concatenation -> dimension = 768 + 5 = 773
    fused_feat = torch.cat([text_emb, structural_feat], dim=0)
    node_features[var] = fused_feat.tolist()

# ---------------------------
# Step 4: Build a simple Variable Dependency Graph (VDG) using NetworkX.
# ---------------------------
G = nx.DiGraph()
for var in vocab:
    G.add_node(var, x=node_features[var])
# Randomly add 15 directed edges between distinct variables.
for _ in range(15):
    u = random.choice(vocab)
    v = random.choice(vocab)
    if u != v:
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1, edge_type="dependency")

print("\nGraph edges:")
for u, v, data in G.edges(data=True):
    print(f"{u} -> {v}, weight: {data['weight']}")

# ---------------------------
# Step 5: Convert the NetworkX graph into a PyTorch Geometric Data object.
# ---------------------------
node_order = list(G.nodes())  # Save node order for later lookup
data = from_networkx(G)
# Create node feature matrix from the fused features.
data.x = torch.tensor([G.nodes[node]['x'] for node in node_order], dtype=torch.float)
print("\nPyTorch Geometric Data object:")
print(data)
print("Edge index:")
print(data.edge_index)

# ---------------------------
# Step 6: Define a simple GCN model with a fallback projection in the encode method.
# ---------------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # Fallback projection: project a 773-dim fused feature to 128-dim (out_channels)
        self.fallback_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """
        Given a variable name or a list of variable names, return the refined embeddings.
        If a variable is in the graph, use its refined GCN output; if not, compute a fused feature
        (textual embedding + zero structural features) and project it via fallback_proj.
        """
        self.eval()
        with torch.no_grad():
            refined = self.forward(data.x, data.edge_index)  # shape: (#nodes, out_channels)
        embeddings = []
        # Ensure inputs is a list.
        if isinstance(inputs, str):
            inputs = [inputs]
        for var in inputs:
            if var in node_order:
                idx = node_order.index(var)
                embeddings.append(refined[idx])
            else:
                # Fallback: compute fused feature = textual embedding concatenated with zeros for structural features.
                text_emb = get_textual_embedding(var)  # shape: (768,)
                fallback_fused = torch.cat([text_emb, torch.zeros(5, dtype=torch.float)], dim=0)  # shape: (773,)
                # Project fallback fused feature to out_channels dimension.
                fallback_embedding = self.fallback_proj(fallback_fused.unsqueeze(0)).squeeze(0)
                embeddings.append(fallback_embedding)
        return torch.stack(embeddings, dim=0)

    def score(self, inputx: Union[str, List[str]], inputy: Union[str, List[str]]) -> List[float]:
        embx = self.encode(inputx)
        emby = self.encode(inputy)
        return F.cosine_similarity(embx, emby).tolist()

# ---------------------------
# Step 7: Train the GCN on the VDG.
# ---------------------------
in_channels = data.x.size(1)  # 773
hidden_channels = 768
out_channels = 768
model_gcn = SimpleGCN(in_channels, hidden_channels, out_channels)
optimizer = torch.optim.Adam(model_gcn.parameters(), lr=0.001)

model_gcn.train()
for epoch in range(10):
    optimizer.zero_grad()
    refined_embeddings = model_gcn(data.x, data.edge_index)
    loss = torch.mean(refined_embeddings ** 2)  # dummy loss for demonstration
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

print("\nFinal refined embeddings:")
print(refined_embeddings)

# ---------------------------
# Step 8: Evaluate on benchmark CSV files.
# ---------------------------
# Build benchmark instance (adjust variant and metric as needed).
b1 = Benchmark.build("idbench", variant="medium", metric="similarity")
b2 = Benchmark.build("idbench", variant="medium", metric="relatedness")
id1_list, id2_list = b1.get_inputs()  # Lists of variable names

# Encode all benchmark pairs using our model.
emb_id1 = model_gcn.encode(id1_list)
emb_id2 = model_gcn.encode(id2_list)
print(emb_id2.shape)
print(emb_id1.shape)
predicted = F.cosine_similarity(emb_id1, emb_id2).tolist()

print("\nBenchmark evaluation (similarity):")
print(b1.evaluate(predicted))
print(b2.evaluate(predicted))
print(b1.evaluate(model.score(id1_list, id2_list)))
# ---------------------------
# Step 9: Use the encode method to obtain refined embeddings for given variables.
# ---------------------------
embeddings_for_vars = model_gcn.encode(["number", "values"])
print("\nRefined embeddings for 'number' and 'values':")
print(embeddings_for_vars)

# ---------------------------
# Step 10: Visualize the Graph and Save as PNG.
# ---------------------------
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Sample Variable Dependency Graph")
plt.axis("off")
plt.savefig("sample_vdg.png")
plt.show()
print("Graph visualization saved as sample_vdg.png")