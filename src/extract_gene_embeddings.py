import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import pickle
import sys

from datasets import Dataset
import pandas as pd
import pyarrow as pa
import torch

sys.path.insert(0, './src/Geneformer')
from geneformer import EmbExtractor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
    )

    parser.add_argument(
        '--ge',
        type=str,
        default='./data/LINCS/processed_level5_beta_trt_cp.tsv',
    )

    parser.add_argument(
        '--sig',
        type=str,
        default='./data/LINCS/processed_siginfo_beta_trt_cp.tsv',
    )

    parser.add_argument(
        '--cell',
        type=str,
        default='./data/LINCS/processed_cellinfo_beta.tsv',
    )

    parser.add_argument(
        '--token',
        type=str,
        default='./data/Geneformer/LINCS_lm_token_95M.csv',
    )

    args = parser.parse_args()
    return args

def load_data(args):
    data = pd.read_table(args.ge, index_col=0)
    df_LINCS_lm_token = pd.read_csv(args.token)

    if (data.columns != df_LINCS_lm_token['gene_name']).sum() == 0:
        data.columns = df_LINCS_lm_token['token']
    else:
        raise ValueError('Gene names in --ge and --token are not matched.')
    
    sig_info = pd.read_table(args.sig, index_col='sig_id')

    if (data.index != sig_info.index).sum() != 0:
        raise ValueError('Signature IDs are not matched.')
    
    cell_info = pd.read_table(args.cell, index_col=0)

    return data, sig_info, cell_info

def make_dataset_file(data, sig_info, cell_info, args):
    dataset = []
    for idx, (sig_id, row) in enumerate(data.iterrows()):
        input_ids = [2] + row.sort_values(ascending=False).index.to_list()
        length = len(input_ids)
        cell_type = sig_info.loc[sig_id, 'cell_iname']
        individual = idx
        age = float(cell_info.loc[cell_type, 'donor_age'])
        sex = cell_info.loc[cell_type, 'donor_sex']
        disease = cell_info.loc[cell_type, 'primary_disease']

        dataset.append([input_ids, length, cell_type, individual, age, sex, disease])

    cols = ['input_ids', 'length', 'cell_type', 'individual', 'age', 'sex', 'disease']
    dataset_df = pd.DataFrame(dataset, columns=cols)

    dataset_all = Dataset(pa.Table.from_pandas(dataset_df))

    dataset_file = f'{args.data_dir}/processed.dataset'
    os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
    dataset_all.save_to_disk(dataset_file)
    return dataset_file

def extract_geneformer_embeddings(dataset_file, args):
    embex = EmbExtractor(model_type='Pretrained', emb_mode='cls', max_ncells=None)

    embs = embex.extract_embs(
        model_directory='./src/Geneformer/gf-12L-95M-i4096',
        input_data_file=dataset_file,
        output_directory=f'{args.data_dir}/extracted_geneformer_embs',
        output_prefix='pretrained'
    )

    embs = embs.type(torch.float16)

    torch.save(embs, f'{args.data_dir}/extracted_geneformer_embs.pt')

def main(args):
    data, sig_info, cell_info = load_data(args)
    dataset_file = make_dataset_file(data, sig_info, cell_info, args)
    extract_geneformer_embeddings(dataset_file, args)

if __name__ == '__main__':
    args = parse_args()
    main(args)