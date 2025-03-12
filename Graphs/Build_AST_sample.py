import json
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

# Load the python grammar (or any other language you're interested in)
print(Parser().language)
# Load the compiled library into the parser
PYTHON_LANGUAGE = Language(tspython.language())
JAVA_LANGUAGE = Language(tsjava.language())
GO_LANGUAGE = Language(tsgo.language())

# Create a parser and set the language
parser_python = Parser(PYTHON_LANGUAGE)
parser_java = Parser(JAVA_LANGUAGE)
parser_go = Parser(GO_LANGUAGE)

print(parser_python.language)
print(parser_java.language)
print(parser_go.language)

# get the identifier from the code
def get_identifier(code, root_node):
    pos = []
    identifiers = []
    def traverse(root):
        if root is None:
            return
        for child in root.children:
            if child.type == 'identifier':
                start_point = child.start_point
                end_point = child.end_point
                pos.insert(0, (start_point, end_point))
            traverse(child)
    traverse(root_node)
    for id in pos:
        identifiers.append(get_identifier_from_position(code, id[0], id[1]))
    # return identifiers, and the position of all identifiers
    return list(set(identifiers)), pos

# get the identifier from the code with the fix position
def get_identifier_from_position(code_string, start_point, end_point):
    lines = code_string.splitlines()
    identifier = lines[start_point[0]][start_point[1]:end_point[1]]
    return identifier


# Example Python code
code = """
def sum(a, b):
    a += 7
    b -= 3
    return a + b
"""

# Parse the code and generate an AST
tree = parser_python.parse(bytes(code, "utf8"))

# Print the tree's root node
print(tree.root_node)

identifiers = get_identifier(code, tree.root_node)
print("Identifiers found:", identifiers)

with open("../data/extracted_go_data.json", 'r', encoding='utf-8') as f:
    go_data = json.load(f)

with open("../data/extracted_python_data.json", 'r', encoding='utf-8') as f:
    python_data = json.load(f)

with open("../data/extracted_java_data.json", 'r', encoding='utf-8') as f:
    java_data = json.load(f)


def print_identifiers(parser, data):
    """make sure the parser language and data language match"""
    for i, sample in enumerate(data[:20]):
        function_code = sample.get("function_code", "")
        if not function_code:
            continue
        print(f"--- Sample {i + 1} ---")
        # Encode the code as bytes since Tree-sitter works with byte offsets.
        tree= parser.parse(bytes(function_code, 'utf-8'))
        identifiers = get_identifier(function_code, tree.root_node)
        print(identifiers)
        print(tree.root_node)

print_identifiers(parser_python, python_data)
print_identifiers(parser_java, java_data)
print_identifiers(parser_go, go_data)
