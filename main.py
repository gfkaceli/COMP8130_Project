from datasets import load_dataset

# where to run and load models

python_ds = load_dataset("claudios/code_search_net", 'python')


python_ds['train'].to_json("data/python_train.json")
python_ds['test'].to_json("data/python_test.json")
python_ds['validation'].to_json("data/python_val.json")

java_ds = load_dataset("claudios/code_search_net", 'java')
