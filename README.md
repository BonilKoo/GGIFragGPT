# GGIFragGPT
GGIFragGPT: transcriptome-conditioned molecule generation via gene-gene interaction-aware fragment modeling

## Installation

### Dependency

The code has been tested in the following environment:

Environment file: `environment.yml`

- Python 3.10.16
- anndata 0.11.3
- cmappy 4.0.1
- datasets 3.4.0
- loompy 3.0.8
- matplotlib 3.10.1
- numpy 1.26.4
- optuna 4.2.1
- optuna-integration[tensorboard] 4.2.1
- pandas 2.2.3
- pyarrow 19.0.1
- peft 0.14.0
- rdkit 2024.9.6
- scanpy 1.11.0
- tdigest 0.5.2.2
- tensorboard 2.19.0
- torch 2.6.0+cu124
- tqdm 4.67.1
- transformers 4.49.0

You can change the package version according to your need.

### Install via Mamba

You can set up the environment using [Mamba](https://github.com/conda-forge/miniforge).
```bash
mamba env create -f environment.yml
mamba activate GGIFragGPT
pip install optuna-integration\[tensorboard\]==4.2.1 --no-deps
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 # check your NVIDIA driver version on your system
```

## Usage

### 1. Gene embedding extraction (Geneformer)
```bash
python src/extract_gene_embeddings.py \
  --data_dir ./data \
  --ge ./data/LINCS/processed_level5_beta_trt_cp.tsv \
  --sig ./data/LINCS/processed_siginfo_beta_trt_cp.tsv \
  --cell ./data/LINCS/processed_cellinfo_beta.tsv \
  --token ./data/Geneformer/LINCS_lm_token_95M.csv
```

*Output*: `./data/extracted_geneformer_embs.pt`

### 2. Training
```bash
python src/run.py train \
  --out_path ./result \
  --dataset_name experiment
```

### 3. Testing
```bash
python src/run.py test \
  --out_path ./result \
  --dataset_name experiment
```

### 4. Generation

#### Step 1: Construct signature for target protein
```bash
python src/construct_shRNA_sig.py \
  --target CDK7
```

#### Step 2: Extract gene embedding
```bash
python src/extract_gene_embeddings.py \
  --data_dir ./data/generation \
  --ge ./data/generation/level5_beta_trt_sh_CDK7.tsv \
  --sig ./data/generation/siginfo_beta_trt_sh_CDK7.tsv \
  --cell ./data/LINCS/processed_cellinfo_beta.tsv \
  --token ./data/Geneformer/LINCS_lm_token_95M.csv
```

#### Step 3: run generation
```bash
python src/run.py generate \
  --out_path ./result \
  --dataset_name experiment \
  --ge_emb ./data/generation/extracted_geneformer_embs.pt \
  --sig_data ./data/generation/siginfo_beta_trt_sh_CDK7.tsv \
  --gen_file generated_shRNA_CDK7.csv \
  --n_mols 1000
```

## Argument Overview

### Common Arguments (shared by all modes)

| Argument | Type | Default | Description |
|------------------|----------|---------------|--------------------------------------------------|
| `--dataset_name` | `str` | `experiment` | Dataset identifier |
| `--out_path` | `str` | `./result` | Output folder |
| `--device` | `str` | `cuda:0` | Device to use (`cpu`, `cuda:0`, etc.) |
| `--seed` | `int` | `42` | Random seed |
| `--batch_size` | `int` | `128` | Batch size |
| `--lr` | `float` | `0.0001` | Learning rate |
| `--d_model` | `int` | `16` | Transformer hidden dimension |
| `--n_heads` | `int` | `8` | Number of attention heads |
| `--d_cell_line` | `int` | `4` | Cell-line embedding dimension |
| `--PE_dropout` | `float` | `0.1` | Positional encoding dropout |
| `--dropout` | `float` | `0.1` | Dropout rate |
| `--d_ff` | `int` | `64` | Feed-forward network dimension |
| `--act_func` | `str` | `relu` | Activation function |
| `--n_layers` | `int` | `6` | Number of decoder layers|

---

### Training-Specific Arguments

| Argument | Type | Default | Description |
|------------------|----------|---------------|--------------------------------------------------|
| `--data_path` | `str` | `./data` | Root directory of dataset |
| `--frag_dict` | `str` | `./data/LINCS/fragment_dict.pkl` | Fragment dictionary path |
| `--val_ratio` | `float` | `0.1` | Validation set ratio |
| `--test_ratio` | `float` | `0.1` | Test set ratio |
| `--epochs` | `int` | `1000` | Number of epochs |
| `--ge_emb` | `str` | `./data/extracted_geneformer_embs.pt` | Gene expression embeddings |
| `--sig_data` | `str` | `./data/LINCS/processed_siginfo_beta_trt_cp.tsv` | Signature metadata |

---

### Generation-Specific Arguments

| Argument | Type | Default | Description |
|------------------|----------|---------------|--------------------------------------------------|
| `--n_mols` | `int` | `1` | Number of molecules to generate |
| `--gen_file` | `str` | `generated.csv` | Output file for SMILES generation |
| `--ge_emb` | `str` | _(required)_ | Gene expression embedding path |
| `--sig_data` | `str` | _(required)_ | Signature TSV file |

---