
# Integrating Graph Neural Networks (GNNs) with Contrastive Learning for Condensed Variable Semantic Representation

This repository implements a self-supervised framework that learns **128‑dimensional** embeddings for program variables by combining local contrastive learning (InfoNCE) with global mutual‑information maximization (Deep Graph Infomax, DGI) on a Variable Dependency Graph. Variables absent from the graph are handled via a lightweight fallback projection of pretrained textual embeddings.

## Repository Structure

The project is organized as follows:

- **Graphs/**  
  Contains everything related to building and visualizing the Variable Dependency Graphs (VDGs):  
  - `Build_VDG_{langauge}.py` — Scripts to construct VDGs from source code in a language.  
  - `text_logs/` — Text logs generated during the graph-building process.  
  - `Visualization/` — Saved graph images .

- **idbench/**  
  Holds the IdBench benchmark CSVs for different agreement thresholds:  
  - `small_pair_wise.csv`  
  - `medium_pair_wise.csv`  
  - `large_pair_wise.csv`

- **my_data/**  
  Preprocessing utilities for raw datasets:  
  - `data_preprocess.py` — Cleans and structures CodeSearchNet data for graph construction.

- **VarCLR/**  
  Pretrained VarCLR model files and helper scripts (used for textual embeddings).

- **COMP8130_Project/**  
  The core GNN training and evaluation code, plus project configuration:  
  - `GNN.py` — Defines the GCN + DGI architecture and embedding utilities.  
  - `main.py` — Loads data, runs training loops, and evaluates on IdBench.  
  - `requirements.txt` — Lists Python dependencies.  
  - `README.md` — This file.
  - `sample_vdg.png` - a sample vdg

## Installation Guide
1. Clone the repository: 
   ```bash
    git clone https://github.com/gfkaceli/COMP8130_Project.git
    cd COMP8130_Project
2. Create and activate a virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate
3. Install Dependencies:
   ```bash
    pip install -r requirements.txt

4. Install VarCLR:
 - to install VarCLR refer to their respective github page, however installation might not be necessary


## Usage

1. To pre-proccess the CodeSearchNet data:
   ```bash
   cd my_data
   python datapreprocess.py
2. To generate a VDG in python, for another language switch python with java or go:
   ```bash
   cd Graphs
   python Build_VDG_python.py
3. To Train the model:
   ```bash
   python main.py

## Features

**128‑Dimensional Embeddings**  
Distills fused textual (768‑dim) + structural (4‑dim) features into a compact 128‑dimensional space.

**Dual Objective Training**  
- **InfoNCE Contrastive Loss** over positive (dependency‑connected) and negative (random) pairs.  
- **Deep Graph Infomax (DGI)** to maximize mutual information between node embeddings and a global summary.

**Fallback Projection**  
Projects out‑of‑graph variable names’ textual embeddings into the same 128‑dimensional latent space.

**Benchmark Evaluation**  
Automatically evaluates on IdBench’s small, medium, and large splits for both similarity and relatedness.

## Configuration
- Hyperparameters (learning rate, epochs, hidden/output dimensions) can be adjusted in main.py.

- GraphML path defaults to Graphs/global_variable_dependency_graph.graphml.

## Results
- Despite using only 128 dimensions, our embeddings recover over 80 percent of the performance of 768‑dimensional baselines (VarCLR, VarGAN). See Tables 1–4 in COMP8130_Final_Project.tex for full benchmarking results:

- Similarity: Spearman up to 0.47, Pearson up to 0.42.

- Relatedness: Spearman up to 0.75, Pearson up to 0.72.

## License
This project is released under the MIT License. Feel free to use and adapt it for your research and applications.