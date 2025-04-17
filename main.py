
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx
from GNN import DGIEncoder, DGI, encode_variable
from torch import nn

def build_data(graphml_path: str, get_text_emb):
    """
    Loads a GraphML, fuses text+structural features, and returns
    a PyG Data object plus the node_order list.
    """
    G = nx.read_graphml(graphml_path)
    node_order = list(G.nodes())

    # Build fused feature dict
    fused = {}
    for n in node_order:
        # structural: 4‑dim from node attribute "x"
        s_str = G.nodes[n].get("x", "0,0,0,0")
        struct = torch.tensor([float(v) for v in s_str.split(",")], dtype=torch.float)
        text = get_text_emb(n)  # 768‑dim
        feat = torch.cat([text, struct], dim=0)  # 772‑dim
        fused[n] = feat
        G.nodes[n]["x"] = feat.tolist()

    data = from_networkx(G)
    data.x = torch.stack([fused[n] for n in node_order], dim=0)
    return data, node_order

def train_dgi(data, in_ch, hid_ch, out_ch, epochs=200, lr=3e-3):
    encoder = DGIEncoder(in_ch, hid_ch, out_ch)
    model = DGI(encoder)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pos, neg = model(data.x, data.edge_index)
        loss = model.loss(pos, neg)
        loss.backward()
        opt.step()
        print(f"Epoch {ep}: DGI Loss = {loss.item():.4f}")

    return model

def evaluate(model, data, node_order):
    # Fallback projection to map 772→128
    fallback = nn.Linear(data.x.size(1), model.encoder.conv2.out_channels)
    fallback.eval()

    model.eval()
    with torch.no_grad():
        H = model.encoder(data.x, data.edge_index)

    from varclr.benchmarks import Benchmark
    benchmarks = [
        Benchmark.build("idbench", variant=v, metric=m)
        for v in ("small","medium","large")
        for m in ("similarity","relatedness")
    ]

    for bench in benchmarks:
        a1, a2 = bench.get_inputs()
        emb1 = torch.stack([encode_variable(n, node_order, H, fallback) for n in a1], dim=0)
        emb2 = torch.stack([encode_variable(n, node_order, H, fallback) for n in a2], dim=0)
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        scores = F.cosine_similarity(emb1, emb2).tolist()
        print(f"{bench.variant}/{bench.metric}: {bench.evaluate(scores)}")

def main():
    torch.manual_seed(42)
    graph_path = "Graphs/global_variable_dependency_graph.graphml"
    from GNN import get_textual_embedding

    data, node_order = build_data(graph_path, get_textual_embedding)
    in_ch = data.x.size(1)    # 772
    model = train_dgi(data, in_ch, hid_ch=128, out_ch=128)
    evaluate(model, data, node_order)

if __name__ == "__main__":
    main()