# GGIFragGPT

## Requirements
- Python 3.10.16
- numpy 1.26.4
- pandas 2.2.3
- rdkit 2024.9.6
- torch 2.6.0+cu124
- tqdm 4.67.1

Environment file:  
```bash
environment.yml
```

## Installation & Setup
```bash
mamba env create -f environment.yml
mamba activate GGIFragGPT
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Usage
Three main workflows are supported:

### 1. Training
```bash
python src/train.py \
  --dataset_name train_dict_unk \
  --device cuda:0 \
  --d_model 16 \
  --d_ff 64
```

### 2. Testing
```bash
python src/test.py \
  --dataset_name train_dict_unk \
  --device cuda:0 \
  --d_model 16 \
  --d_ff 64
```

### 3. Generation
```bash
python src/generate.py \
  --ge_emb data/processed_level5_beta_trt_sh_CDK7.pt \
  --sig_data data/processed_siginfo_beta_trt_sh_CDK7.tsv \
  --gen_file sh_CDK7.csv \
  --n_mols 1000 \
  --dataset_name train_dict_unk \
  --device cuda:0 \
  --d_model 16 \
  --d_ff 64 \
  --data_path ../../result
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