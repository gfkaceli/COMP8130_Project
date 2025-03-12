from datasets import load_dataset
import json
# preprocesses the data

python_ds = load_dataset("claudios/code_search_net", 'python')
java_ds = load_dataset("claudios/code_search_net", 'java')
go_ds = load_dataset("claudios/code_search_net", "go")

def extract_function_data(dataset, doc_key="func_documentation_string", func_key="func_code_string"):
    extracted = []
    for example in dataset:
        doc = example.get(doc_key, "").strip()
        func_str = example.get(func_key, "").strip()
        extracted.append({
            "documentation": doc,
            "function_code": func_str
        })
    return extracted

# Extract data from the test split (you can choose a different split)
extracted_data_python = extract_function_data(python_ds["test"])
extracted_data_java = extract_function_data(java_ds["test"])
extracted_data_go = extract_function_data(go_ds["test"])

# Convert the extracted data to a JSON string and write it to a file
output_file = "extracted_python_data.json"
with open(output_file, "w", encoding="utf-8") as fout:
    json.dump(extracted_data_python, fout, indent=2)

# Convert the extracted data to a JSON string and write it to a file
output_file2 = "extracted_java_data.json"
with open(output_file2, "w", encoding="utf-8") as fout:
    json.dump(extracted_data_java, fout, indent=2)

# Convert the extracted data to a JSON string and write it to a file
output_file3 = "extracted_go_data.json"
with open(output_file3, "w", encoding="utf-8") as fout:
    json.dump(extracted_data_go, fout, indent=2)



