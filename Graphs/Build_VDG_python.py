import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import torch
from torch_geometric.utils import from_networkx
import logging

# ---------------------------
# Setup Logging
# ---------------------------
log_dir = "text_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "build_vdg_python_log.txt")
logging.basicConfig(filename=log_file,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()
logger.info("Starting VDG building process...")

# ---------------------------
# Setup: Load Python grammar using your current Tree-sitter format.
# ---------------------------
logger.info("Loading Python grammar using tree-sitter...")
PYTHON_LANGUAGE = Language(tspython.language())
parser = Parser(PYTHON_LANGUAGE)
logger.info("Python grammar loaded.\n")

# Global counters for frequency and co-occurrence.
global_frequency = Counter()
co_occurrence_counter = Counter()

# Define threshold for high frequency.
HIGH_FREQ_THRESHOLD = 5

# ---------------------------
# Helper Functions for AST Parsing and Variable Extraction
# ---------------------------
def get_identifier_from_position(code_string, start_point, end_point):
    """Extract the identifier text from code using node positions (assumes the identifier is on one line)."""
    lines = code_string.splitlines()
    return lines[start_point[0]][start_point[1]:end_point[1]].strip()

def extract_variable_names(node, code):
    """Recursively extract unique variable names from AST nodes."""
    names = []
    if node is None:
        return names
    if node.type == "identifier":
        name = get_identifier_from_position(code, node.start_point, node.end_point)
        names.append(name)
    elif node.type == "attribute":
        # For attribute nodes, return only the base object's name.
        object_node = node.child_by_field_name("object")
        names.extend(extract_variable_names(object_node, code))
    elif node.type == "subscript":
        value_node = node.child_by_field_name("value")
        names.extend(extract_variable_names(value_node, code))
    elif node.type in ["tuple", "list"]:
        for child in node.children:
            if child.type != ",":
                names.extend(extract_variable_names(child, code))
    elif node.type == "call":
        function_node = node.child_by_field_name("function")
        if function_node and function_node.type == "attribute":
            object_node = function_node.child_by_field_name("object")
            names.extend(extract_variable_names(object_node, code))
        else:
            arguments_node = node.child_by_field_name("arguments")
            if arguments_node:
                names.extend(extract_variable_names(arguments_node, code))
    else:
        for child in node.children:
            names.extend(extract_variable_names(child, code))
    return list(set(names))

def extract_assignment_edges(root_node, code):
    """
    Extract assignment edges from the AST.
    For a simple assignment like: x = y, add an edge from y to x.
    For a call-based assignment such as: x = y.functioncall(z),
    add an edge from each argument (z) to x.
    """
    edges = []

    def process_assignment(node):
        lhs_node = node.child_by_field_name("left")
        rhs_node = node.child_by_field_name("right")
        if lhs_node is None or rhs_node is None:
            if len(node.children) >= 2:
                lhs_node = node.children[0]
                rhs_node = node.children[-1]
            else:
                return
        lhs_vars = extract_variable_names(lhs_node, code)
        if rhs_node.type == "call":
            function_node = rhs_node.child_by_field_name("function")
            arguments_node = rhs_node.child_by_field_name("arguments")
            # Special case: if it is a method call with arguments, map each argument to the LHS.
            if function_node and function_node.type == "attribute" and arguments_node:
                arg_vars = extract_variable_names(arguments_node, code)
                for arg in arg_vars:
                    for l in lhs_vars:
                        if arg != l:
                            edges.append((arg, l))
            else:
                if function_node and function_node.type == "attribute":
                    object_node = function_node.child_by_field_name("object")
                    base_vars = extract_variable_names(object_node, code)
                    for base in base_vars:
                        for l in lhs_vars:
                            if base != l:
                                edges.append((base, l))
                else:
                    rhs_vars = extract_variable_names(arguments_node, code)
                    for l in lhs_vars:
                        for r in rhs_vars:
                            if l != r:
                                edges.append((l, r))
        else:
            rhs_vars = extract_variable_names(rhs_node, code)
            for l in lhs_vars:
                for r in rhs_vars:
                    if l != r:
                        edges.append((r, l))

    def traverse(node):
        if node.type in ["assignment", "augmented_assignment"]:
            process_assignment(node)
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return edges

def extract_usage_edges(root_node, code):
    """
    Extract usage edges from function definitions:
    For each function, extract its name and then from return statements
    extract variables used in returns, adding an edge from each variable to the function's name.
    """
    usage_edges = []
    def traverse(node, func_name=None):
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = get_identifier_from_position(code, name_node.start_point, name_node.end_point)
        if node.type == "return_statement" and func_name:
            ret_vars = extract_variable_names(node, code)
            for var in ret_vars:
                if var != func_name:
                    usage_edges.append((var, func_name))
        for child in node.children:
            traverse(child, func_name)
    traverse(root_node, None)
    return usage_edges

# ---------------------------
# Integrated VDG Construction Across the Dataset
# ---------------------------
dataset_file = "../my_data/extracted_python_data.json"
with open(dataset_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

global_vdg = nx.DiGraph()

logger.info("Processing samples and building integrated VDG...")
sample_count = 0
for sample in samples[950:1000]:  # Change sample count as desired.
    code = sample.get("function_code", "")
    if not code.strip():
        continue
    sample_count += 1
    logger.info(f"Processing sample {sample_count}...")
    tree = parser.parse(bytes(code, "utf-8"))
    root_node = tree.root_node

    sample_assign_edges = extract_assignment_edges(root_node, code)
    logger.info(f"  Assignment edges: {sample_assign_edges}")

    sample_usage_edges = extract_usage_edges(root_node, code)
    logger.info(f"  Usage edges: {sample_usage_edges}")

    sample_edges = sample_assign_edges + sample_usage_edges
    if not sample_edges:
        logger.info("  No dependency edges found; skipping sample.")
        continue

    edge_nodes = set()
    for edge in sample_edges:
        edge_nodes.update(edge)
    logger.info(f"  Variables in dependency edges: {edge_nodes}")

    for var in edge_nodes:
        global_vdg.add_node(var)
        global_frequency[var] += 1

    for v1, v2 in combinations(sorted(edge_nodes), 2):
        co_occurrence_counter[(v1, v2)] += 1

    for edge in sample_edges:
        if global_vdg.has_edge(*edge):
            global_vdg[edge[0]][edge[1]]["weight"] = global_vdg[edge[0]][edge[1]].get("weight", 1) + 1
            if edge in sample_usage_edges:
                global_vdg[edge[0]][edge[1]]["edge_type"] = "usage"
        else:
            edge_type = "usage" if edge in sample_usage_edges else "dependency"
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
# Compute simplified structural features for each node.
# Features: [frequency, co-occurrence sum, length, high frequency flag]
# ---------------------------
logger.info("\nComputing simplified graph structural features...")
for node in global_vdg.nodes():
    frequency_feature = global_frequency.get(node, 0)
    co_occurrence_sum = sum(count for (n1, n2), count in co_occurrence_counter.items() if node in (n1, n2))
    length_feature = len(node)
    high_freq_flag = 1 if frequency_feature >= HIGH_FREQ_THRESHOLD else 0
    new_features = [frequency_feature, co_occurrence_sum, length_feature, high_freq_flag]
    global_vdg.nodes[node]['x'] = new_features

logger.info("Updated node features with simplified structural metrics:")
for node, attr in global_vdg.nodes(data=True):
    logger.info(f"{node}: {attr['x']}")

# ---------------------------
# Before saving, convert node feature lists to comma-separated strings.
# ---------------------------
for node in global_vdg.nodes():
    if isinstance(global_vdg.nodes[node].get('x'), list):
        feature_list = global_vdg.nodes[node]['x']
        global_vdg.nodes[node]['x'] = ",".join(map(str, feature_list))

# ---------------------------
# Save the global VDG as a GraphML file.
# ---------------------------
output_graphml_file = "global_variable_dependency_graph.graphml"
nx.write_graphml(global_vdg, output_graphml_file)
logger.info(f"\nGlobal VDG saved as GraphML: {output_graphml_file}")

# ---------------------------
# Convert the global VDG to a PyTorch Geometric Data object.
# ---------------------------
logger.info("\nConverting global VDG to PyTorch Geometric Data object...")
data = from_networkx(global_vdg)
# Check if data.x is a tensor. If not, assume it's a list of strings and convert.
if not isinstance(data.x, torch.Tensor):
    # data.x is expected to be a list of node features stored as comma-separated strings.
    # We want to convert each string back into a list of floats.
    new_x = []
    for feature in data.x:
        if isinstance(feature, str):
            # Split the string by commas, strip spaces, and convert each token to float.
            float_list = [float(x.strip()) for x in feature.split(",") if x.strip() != ""]
            new_x.append(float_list)
        else:
            new_x.append(feature)
    data.x = torch.tensor(new_x, dtype=torch.float)
else:
    data.x = data.x.clone().detach().float()
logger.info("PyTorch Geometric Data object:")
logger.info(data)
logger.info("Edge index:")
logger.info(data.edge_index)

# ---------------------------
# Visualize the Graph and Save as PNG.
# ---------------------------
logger.info("\nVisualizing the graph...")
pos = nx.spring_layout(global_vdg, seed=42)
plt.figure(figsize=(12, 10))
dep_edges = [(u, v) for u, v, d in global_vdg.edges(data=True)
             if d.get('edge_type') in ["dependency", "dependency+virtual", "usage"] and d.get("virtual_weight", 1) == 1]
virtual_edges = [(u, v) for u, v, d in global_vdg.edges(data=True)
                 if d.get("virtual_weight", 1) > 1 or d.get('edge_type') == "virtual"]
nx.draw_networkx_nodes(global_vdg, pos, node_color="skyblue", node_size=500)
nx.draw_networkx_edges(global_vdg, pos, edgelist=dep_edges, edge_color='blue', arrowstyle='->', arrowsize=10, width=2)
nx.draw_networkx_edges(global_vdg, pos, edgelist=virtual_edges, edge_color='red', arrowstyle='->', arrowsize=10,
                       width=2, style='dashed')
nx.draw_networkx_labels(global_vdg, pos, font_size=10, font_family="sans-serif")
plt.title("Global Variable Dependency Graph")
plt.axis("off")
plt.savefig("Visualization/global_python_variable_dependency_graph.png")
plt.show()
logger.info("Graph visualization saved to 'Visualization/global_python_variable_dependency_graph.png'.")
