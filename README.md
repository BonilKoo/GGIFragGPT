# GGIFragGPT
GGIFragGPT: transcriptome-conditioned molecule generation via gene-gene interaction-aware fragment modeling

## Installation

### Dependency

The code has been tested in the following environment:

Environment file:  
`environment.yml`

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
pip install optuna-integration[tensorboard]==4.2.1 --no-deps
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Usage

### 1. Gene embedding extraction using Geneformer
```bash
python src/Geneformer_extract_embeddings.py \
  --data_dir ./data \
  --ge ./data/LINCS/processed_level5_beta_trt_cp.tsv \
  --sig ./data/LINCS/processed_siginfo_beta_trt_cp.tsv \
  --cell ./data/LINCS/processed_cellinfo_beta.tsv \
  --token ./data/Geneformer/LINCS_lm_token_95M.csv
```

*Output*
- `./data/extracted_geneformer_embs.pt`

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

#### Step 1
```bash
python src/construct_shRNA_sig.py \
  --target CDK7
```

#### Step 2
```bash
python src/extract_gene_embeddings.py \
  --data_dir ./data/generation \
  --ge ./data/generation/level5_beta_trt_sh_CDK7.tsv \
  --sig ./data/generation/siginfo_beta_trt_sh_CDK7.tsv \
  --cell ./data/LINCS/processed_cellinfo_beta.tsv \
  --token ./data/Geneformer/LINCS_lm_token_95M.csv
```

#### Step 3
```bash
python src/run.py generate \
  --out_path ./result \
  --dataset_name experiment \
  --ge_emb ./data/generation/processed_level5_beta_trt_sh_CDK7.pt \
  --sig_data ./data/generation/processed_siginfo_beta_trt_sh_CDK7.tsv \
  --gen_file generated_shRNA_CDK7.csv \
  --n_mols 1000
```

#### Common Arguments
- `--dataset_name` (str, default: `test`): Name of the dataset to use.
- `--device` (str, default: `cuda:0`): Device for computation (`cpu` or `cuda:0`, etc.).
- `--seed` (int, default: 42): Random seed for reproducibility.
- `--batch_size` (int, default: 128): Batch size for training (pretraining may use 512).
- `--lr` (float, default: 0.0001): Learning rate.
- `--d_model` (int, default: 16): Hidden dimension size of the transformer.
- `--n_heads` (int, default: 8): Number of attention heads.
- `--d_cell_line` (int, default: 4): Dimension of the cell-line embedding.
- `--PE_dropout` (float, default: 0.1): Dropout rate for positional encoding.
- `--dropout` (float, default: 0.1): General dropout rate.
- `--d_ff` (int, default: 64): Dimension of the feed-forward layer.
- `--act_func` (str, default: `relu`): Activation function.
- `--n_layers` (int, default: 6): Number of transformer decoder layers.

#### Dataset & Data-Related Arguments
- `--data_path` (str, default: `../../data`): Root path for all data files.
- `--frag_dict` (str, default: `../../data/LINCS/fragment/fragment_dict.pkl`): Fragment dictionary (pickle).
- `--val_ratio` (float, default: 0.1): Validation split ratio.
- `--test_ratio` (float, default: 0.1): Test split ratio.
- `--epochs` (int, default: 1000): Number of training epochs.

#### Fine-Tuning & Pretrained Settings
- `--frag_dict_pretrained` (str, default: `../../data/ChEMBL/fragment/fragment_dict.pkl`): Pretrained fragment dictionary.
- `--pretrained_ckpts` (str, default: `../../result/pretrain/ckpts_test/dim128_n6h8ff512_bs512_lr0.0001/best_model.ckpt`): Path to pretrained checkpoint.

#### Generation & Attention Extraction
- `--n_mols` (int, default: 1): Number of molecules to generate.
- `--gen_file` (str, default: `generated.csv`): Output file for generated SMILES.
- `--ge_emb` (str): Path to gene expression embedding file.
- `--sig_data` (str): Path to signature TSV file.
- `--geneformer_dataset` (str): Path to Geneformer dataset TSV.
- `--geneformer_token` (str): Path to Geneformer token CSV.
- `--attn_out` (str): Output path for attention results TSV.

For detailed options and advanced workflows, see the docstrings in `src/args.py`.