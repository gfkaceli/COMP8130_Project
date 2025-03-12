import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava  # Assumed module for Java grammar
import torch
from torch_geometric.utils import from_networkx
import logging

# ---------------------------
# Setup Logging
# ---------------------------
log_dir = "text_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "build_vdg_java_log.txt")
logging.basicConfig(filename=log_file,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()
logger.info("Starting Java VDG building process...")

# ---------------------------
# Setup: Load Java grammar using Tree-sitter.
# ---------------------------
logger.info("Loading Java grammar using tree-sitter...")
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)
logger.info("Java grammar loaded.\n")

# Global counters for variable frequency and co-occurrence.
global_frequency = Counter()
co_occurrence_counter = Counter()


# ---------------------------
# Helper Functions for AST Parsing and Variable Extraction (Java)
# ---------------------------
def get_identifier_from_position(code_string, start_point, end_point):
    """Extract identifier text from a code string using node positions (assumes identifier on one line)."""
    lines = code_string.splitlines()
    return lines[start_point[0]][start_point[1]:end_point[1]].strip()


def extract_variable_names(node, code):
    """
    Recursively extract unique variable names from the Java AST.
    This function collects identifiers from nodes of type "identifier".
    """
    names = []
    if node is None:
        return names
    if node.type == "identifier":
        name = get_identifier_from_position(code, node.start_point, node.end_point)
        names.append(name)
    else:
        for child in node.children:
            names.extend(extract_variable_names(child, code))
    return list(set(names))


def extract_assignment_edges(root_node, code):
    """
    Extract assignment edges from a Java AST.
    For a local variable declaration:
      e.g., local_variable_declaration -> variable_declarator (with fields "name" and "value")
    add an edge from the right-hand side (initializer) to the declared variable name.

    Optionally, you could extend this to assignment expressions.
    """
    edges = []

    def process_local_var_decl(node):
        # In Java, a local_variable_declaration node usually contains one or more variable_declarator children.
        for child in node.children:
            if child.type == "variable_declarator":
                # Extract the variable name.
                name_node = child.child_by_field_name("name")
                # Extract the initializer (value) if it exists.
                value_node = child.child_by_field_name("value")
                if name_node and value_node:
                    lhs_vars = extract_variable_names(name_node, code)
                    rhs_vars = extract_variable_names(value_node, code)
                    for r in rhs_vars:
                        for l in lhs_vars:
                            if r != l:
                                edges.append((r, l))

    def process_assignment_expr(node):
        # Optional: Process nodes of type "assignment_expression"
        # Typically, these have "left" and "right" fields.
        lhs_node = node.child_by_field_name("left")
        rhs_node = node.child_by_field_name("right")
        if lhs_node and rhs_node:
            lhs_vars = extract_variable_names(lhs_node, code)
            rhs_vars = extract_variable_names(rhs_node, code)
            for r in rhs_vars:
                for l in lhs_vars:
                    if r != l:
                        edges.append((r, l))

    def traverse(node):
        if node.type == "local_variable_declaration":
            process_local_var_decl(node)
        elif node.type == "assignment_expression":
            process_assignment_expr(node)
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return edges


def extract_usage_edges(root_node, code):
    """
    (Optional) Extract usage edges from method declarations.
    For each method_declaration, capture its name and then, from within its body,
    if there is a return_statement, extract variables used in the return expression,
    and add a usage edge from each such variable to the method's name.
    """
    usage_edges = []

    def traverse(node, current_method=None):
        if node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                current_method = get_identifier_from_position(code, name_node.start_point, name_node.end_point)
        if node.type == "return_statement" and current_method:
            ret_vars = extract_variable_names(node, code)
            for var in ret_vars:
                if var != current_method:
                    usage_edges.append((var, current_method))
        for child in node.children:
            traverse(child, current_method)

    traverse(root_node, None)
    return usage_edges


# ---------------------------
# Integrated VDG Construction Across the Java Dataset
# ---------------------------
dataset_file = "../data/extracted_java_data.json"
with open(dataset_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

global_vdg = nx.DiGraph()

logger.info("Processing samples and building integrated Java VDG...")
sample_count = 0
for sample in samples[:500]:  # Process first 20 samples for debugging.
    code = sample.get("function_code", "")
    if not code.strip():
        continue
    sample_count += 1
    logger.info(f"Processing sample {sample_count}...")
    tree = parser.parse(bytes(code, "utf-8"))
    root_node = tree.root_node

    # Extract assignment edges from the Java code.
    assign_edges = extract_assignment_edges(root_node, code)
    logger.info(f"  Assignment edges: {assign_edges}")

    # Optionally, extract usage edges from method declarations.
    usage_edges = extract_usage_edges(root_node, code)
    logger.info(f"  Usage edges: {usage_edges}")

    sample_edges = assign_edges + usage_edges
    if not sample_edges:
        logger.info("  No dependency edges found; skipping sample.")
        continue

    # Determine nodes involved in dependency edges.
    edge_nodes = set()
    for edge in sample_edges:
        edge_nodes.update(edge)
    logger.info(f"  Variables in dependency edges: {edge_nodes}")

    # Update global frequency and add nodes (only for variables in dependency edges).
    for var in edge_nodes:
        global_vdg.add_node(var)
        global_frequency[var] += 1

    # Update co-occurrence statistics for variables in dependency edges.
    for v1, v2 in combinations(sorted(edge_nodes), 2):
        co_occurrence_counter[(v1, v2)] += 1

    # Add dependency edges to the global graph.
    for edge in sample_edges:
        if global_vdg.has_edge(*edge):
            global_vdg[edge[0]][edge[1]]["weight"] = global_vdg[edge[0]][edge[1]].get("weight", 1) + 1
            if edge in usage_edges:
                global_vdg[edge[0]][edge[1]]["edge_type"] = "usage"
        else:
            edge_type = "usage" if edge in usage_edges else "dependency"
            global_vdg.add_edge(*edge, weight=1, edge_type=edge_type)

logger.info("Finished processing samples.\n")
logger.info("Global variable frequencies:")
for var, freq in global_frequency.items():
    logger.info(f"{var}: {freq}")

logger.info("\nGlobal Variable Dependency Graph edges (before virtual edges):")
for u, v, attr in global_vdg.edges(data=True):
    logger.info(f"{u} -> {v} (edge_type: {attr.get('edge_type')}, weight: {attr.get('weight')})")

# ---------------------------
# Add Virtual Edges based on Co-occurrence Statistics.
# ---------------------------
virtual_threshold = 2
logger.info(f"\nAdding virtual edges based on co-occurrence (threshold = {virtual_threshold})...")
for (v1, v2), count in co_occurrence_counter.items():
    if count >= virtual_threshold:
        if global_vdg.has_edge(v1, v2):
            global_vdg[v1][v2]["virtual_weight"] = global_vdg[v1][v2].get("virtual_weight", 1) + count
            if global_vdg[v1][v2]["edge_type"] != "usage":
                global_vdg[v1][v2]["edge_type"] = "dependency+virtual"
        else:
            global_vdg.add_edge(v1, v2, edge_type="virtual", virtual_weight=count)
        if global_vdg.has_edge(v2, v1):
            global_vdg[v2][v1]["virtual_weight"] = global_vdg[v2][v1].get("virtual_weight", 1) + count
            if global_vdg[v2][v1]["edge_type"] != "usage":
                global_vdg[v2][v1]["edge_type"] = "dependency+virtual"
        else:
            global_vdg.add_edge(v2, v1, edge_type="virtual", virtual_weight=count)

# Normalize edge attributes so every edge has the same keys.
logger.info("\nNormalizing edge attributes for consistency...")
for u, v, d in global_vdg.edges(data=True):
    new_d = {
        "virtual_weight": d.get("virtual_weight", 1),
        "edge_type": d.get("edge_type", "dependency")
    }
    global_vdg[u][v].clear()
    global_vdg[u][v].update(new_d)

logger.info("\nGlobal Variable Dependency Graph edges (after adding virtual edges):")
for u, v, attr in global_vdg.edges(data=True):
    logger.info(f"{u} -> {v} (edge_type: {attr.get('edge_type')}, virtual_weight: {attr['virtual_weight']})")

# ---------------------------
# Compute Graph Structural Features and Update Node Features.
# ---------------------------
logger.info("\nComputing graph structural features...")
in_deg = dict(global_vdg.in_degree())
out_deg = dict(global_vdg.out_degree())
total_deg = dict(global_vdg.degree())
betweenness = nx.betweenness_centrality(global_vdg)
pagerank = nx.pagerank(global_vdg)

for node in global_vdg.nodes():
    length_feature = len(node)
    frequency_feature = global_frequency.get(node, 0)
    new_features = [length_feature, frequency_feature,
                    in_deg.get(node, 0), out_deg.get(node, 0),
                    total_deg.get(node, 0), betweenness.get(node, 0),
                    pagerank.get(node, 0)]
    global_vdg.nodes[node]['x'] = new_features

logger.info("Updated node features with graph structural metrics:")
for node, attr in global_vdg.nodes(data=True):
    logger.info(f"{node}: {attr['x']}")

# ---------------------------
# Convert the global VDG to a PyTorch Geometric Data object.
# ---------------------------
logger.info("\nConverting global VDG to PyTorch Geometric Data object...")
data = from_networkx(global_vdg)
if not isinstance(data.x, torch.Tensor):
    data.x = torch.tensor(data.x, dtype=torch.float)
else:
    data.x = data.x.clone().detach().float()

logger.info("PyTorch Geometric Data object:")
logger.info(data)
logger.info("Edge index:")
logger.info(data.edge_index)

# ---------------------------
# Visualize the Graph and Save as PNG.
# ---------------------------
logger.info("Visualizing the graph...")
pos = nx.spring_layout(global_vdg, seed=42)
plt.figure(figsize=(12, 10))

dep_edges = [(u, v) for u, v, d in global_vdg.edges(data=True)
             if d.get('edge_type') in ["dependency", "dependency+virtual", "usage"] and d.get("virtual_weight", 1) == 1]
virtual_edges = [(u, v) for u, v, d in global_vdg.edges(data=True)
                 if d.get("virtual_weight", 1) > 1 or d.get('edge_type') == "virtual"]

nx.draw_networkx_nodes(global_vdg, pos, node_color="lightblue", node_size=500)
nx.draw_networkx_edges(global_vdg, pos, edgelist=dep_edges, edge_color='blue', arrowstyle='->', arrowsize=10, width=2)
nx.draw_networkx_edges(global_vdg, pos, edgelist=virtual_edges, edge_color='red', arrowstyle='->', arrowsize=10,
                       width=2, style='dashed')
nx.draw_networkx_labels(global_vdg, pos, font_size=10, font_family="sans-serif")

plt.title("Global Java Variable Dependency Graph")
plt.axis("off")
plt.savefig("Visualization/global_java_variable_dependency_graph.png")
plt.show()
logger.info("Graph visualization saved to 'global_java_variable_dependency_graph.png'.")