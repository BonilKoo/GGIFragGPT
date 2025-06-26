import os
import pathlib
import pickle
import ast
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, data_path, dataset_name, ge_file, sig_file, frag2idx_file):
        self.data_path = pathlib.Path(data_path)
        self.dataset_name = pathlib.Path(dataset_name)
        self.ge_file = ge_file
        self.sig_file = sig_file
        self.frag2idx_file = frag2idx_file

        self.processed_file_name = (self.data_path / self.dataset_name).with_suffix('.pt')
        
        if os.path.exists(self.processed_file_name):
            (
                self.ge_emb,
                self.cell_lines,
                self.frags_tokens,
                self.frag2idx,
                self.idx2frag,
                self.cell2idx,
                self.max_len,
                self.vocab_size,
                self.padding_idx
            ) = torch.load(self.processed_file_name)
        else:
            self.process()
        self.ge_emb = self.ge_emb.type(torch.float)

    def process(self):
        self.ge_emb = torch.load(self.ge_file)
        
        sig_info = pd.read_table(self.sig_file, index_col=0)
        sig_info['fragments'] = [ast.literal_eval(frags) for frags in sig_info['fragments']]
        
        with open(self.frag2idx_file, 'rb') as f:
            self.frag2idx = pickle.load(f)
        self.idx2frag = {v: k for k, v in self.frag2idx.items()}

        self.cell2idx = {cell_line: idx for idx, cell_line in enumerate(sig_info['cell_iname'].unique())}
        self.cell_lines = torch.from_numpy(np.array([self.cell2idx[cell_line] for cell_line in sig_info['cell_iname']], dtype=np.int64))

        self.max_len = max([len(frags) for frags in sig_info['fragments']])
        self.vocab_size = len(self.frag2idx)
        self.padding_idx = self.frag2idx['[PAD]']
        
        frags_tokens = []
        for frags in sig_info['fragments']:
            frag_tokens = [self.frag2idx['[START]']]
            for frag in frags:
                frag_tokens.append(self.frag2idx[frag])
            frag_tokens.append(self.frag2idx['[END]'])
            for _ in range(self.max_len - len(frags)):
                frag_tokens.append(self.padding_idx)
            frags_tokens.append(frag_tokens)
        
        self.frags_tokens = torch.from_numpy(np.array(frags_tokens, np.int64))
        
        self.max_len += 2 # [STRAT] [END]

        torch.save((
            self.ge_emb,
            self.cell_lines,
            self.frags_tokens,
            self.frag2idx,
            self.idx2frag,
            self.cell2idx,
            self.max_len,
            self.vocab_size,
            self.padding_idx,
        ),
                   self.processed_file_name)

    def __getitem__(self, index):
        return self.ge_emb[index], self.cell_lines[index], self.frags_tokens[index]

    def __len__(self):
        return len(self.cell_lines)

class TestDataset(Dataset):
    def __init__(self, data_path, dataset_name):
        self.data_path = pathlib.Path(data_path)
        self.dataset_name = pathlib.Path(dataset_name)
        
        self.process()
        self.ge_emb = self.ge_emb.type(torch.float)

    def process(self):
        dir_dataset = f'{self.data_path}/{self.dataset_name}/dataset'
        test_data = torch.load(f'{dir_dataset}/test_data.pt', weights_only=False)
        with open(f'{dir_dataset}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        self.ge_emb = test_data['ge_emb']
        
        self.sig_info = test_data['sig_info']
        self.sig_info['fragments'] = [ast.literal_eval(frags) for frags in self.sig_info['fragments']]
        
        self.frag2idx = metadata['frag2idx']
        self.idx2frag = metadata['idx2frag']

        self.cell2idx = metadata['cell2idx']
        self.cell_lines = test_data['cell_lines']

        self.max_len = metadata['max_len']
        self.vocab_size = metadata['vocab_size']
        self.padding_idx = metadata['padding_idx']
        
        self.frags_tokens = test_data['frags_tokens']

    def __getitem__(self, index):
        # return self.ge_emb[index], self.cell_lines[index], self.frags_tokens[index]
        return self.ge_emb[index], self.cell_lines[index]

    def __len__(self):
        return len(self.cell_lines)

class ChEMBLDataset(Dataset):
    def __init__(self, data_path, dataset_name, sig_file, frag2idx_file):
        self.data_path = pathlib.Path(data_path)
        self.dataset_name = pathlib.Path(dataset_name)
        self.sig_file = sig_file
        self.frag2idx_file = frag2idx_file
        
        self.process()

    def process(self):        
        sig_info = pd.read_table(self.sig_file, index_col=0)
        sig_info['fragments'] = [ast.literal_eval(frags) for frags in sig_info['fragments']]
        
        with open(self.frag2idx_file, 'rb') as f:
            self.frag2idx = pickle.load(f)
        self.idx2frag = {v: k for k, v in self.frag2idx.items()}

        self.max_len = max([len(frags) for frags in sig_info['fragments']])
        self.vocab_size = len(self.frag2idx)
        self.padding_idx = self.frag2idx['[PAD]']
        
        frags_tokens = []
        for frags in sig_info['fragments']:
            frag_tokens = [self.frag2idx['[START]']]
            for frag in frags:
                frag_tokens.append(self.frag2idx[frag])
            frag_tokens.append(self.frag2idx['[END]'])
            for _ in range(self.max_len - len(frags)):
                frag_tokens.append(self.padding_idx)
            frags_tokens.append(frag_tokens)
        
        self.frags_tokens = torch.from_numpy(np.array(frags_tokens, np.int64))
        
        self.max_len += 2 # [STRAT] [END]

    def __getitem__(self, index):
        return self.frags_tokens[index]

    def __len__(self):
        return len(self.frags_tokens)

class AttentionDataset(Dataset):
    def __init__(self, data_path, dataset_name, ge_file, sig_file):
        self.data_path = pathlib.Path(data_path)
        self.dataset_name = pathlib.Path(dataset_name)
        self.ge_file = ge_file
        self.sig_file = sig_file
        
        self.process()
        self.ge_emb = self.ge_emb.type(torch.float)

    def process(self):
        dir_dataset = f'{self.data_path}/{self.dataset_name}/dataset'
        test_data = torch.load(f'{dir_dataset}/test_data.pt', weights_only=False)
        with open(f'{dir_dataset}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            
        self.ge_emb = torch.load(self.ge_file)
        
        sig_info = pd.read_table(self.sig_file, index_col=0)
        sig_info['fragments'] = [ast.literal_eval(frags) for frags in sig_info['fragments']]
        
        # with open(self.frag2idx_file, 'rb') as f:
        #     self.frag2idx = pickle.load(f)
        self.frag2idx = metadata['frag2idx']
        # self.idx2frag = {v: k for k, v in self.frag2idx.items()}
        self.idx2frag = metadata['idx2frag']

        # self.cell2idx = {cell_line: idx for idx, cell_line in enumerate(sig_info['cell_iname'].unique())}
        self.cell2idx = metadata['cell2idx']
        self.cell_lines = torch.from_numpy(np.array([self.cell2idx[cell_line] for cell_line in sig_info['cell_iname']], dtype=np.int64))

        # self.max_len = max([len(frags) for frags in sig_info['fragments']])
        self.max_len = metadata['max_len'] - 2
        # self.vocab_size = len(self.frag2idx)
        self.vocab_size = metadata['vocab_size']
        # self.padding_idx = self.frag2idx['[PAD]']
        self.padding_idx = metadata['padding_idx']
        
        frags_tokens = []
        for frags in sig_info['fragments']:
            frag_tokens = [self.frag2idx['[START]']]
            for frag in frags:
                if frag in self.frag2idx:
                    frag_tokens.append(self.frag2idx[frag])
                else:
                    frag_tokens.append(self.frag2idx['[UNK]'])
            frag_tokens.append(self.frag2idx['[END]'])
            for _ in range(self.max_len - len(frags)):
                frag_tokens.append(self.padding_idx)
            frags_tokens.append(frag_tokens)
        
        self.frags_tokens = torch.from_numpy(np.array(frags_tokens, np.int64))
        
        self.max_len += 2 # [STRAT] [END]

    def __getitem__(self, index):
        return self.ge_emb[index], self.cell_lines[index], self.frags_tokens[index]

    def __len__(self):
        return len(self.cell_lines)

class GenerationDataset(Dataset):
    def __init__(self, data_path, dataset_name, ge_file, sig_file):
        self.data_path = pathlib.Path(data_path)
        self.dataset_name = pathlib.Path(dataset_name)
        self.ge_file = ge_file
        self.sig_file = sig_file
        
        self.process()
        self.ge_emb = self.ge_emb.type(torch.float)

    def process(self):
        dir_dataset = f'{self.data_path}/{self.dataset_name}/dataset'
        with open(f'{dir_dataset}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        self.ge_emb = torch.load(self.ge_file)
        
        self.sig_info = pd.read_table(self.sig_file)
        # self.sig_info['fragments'] = [ast.literal_eval(frags) for frags in self.sig_info['fragments']]
        
        self.frag2idx = metadata['frag2idx']
        self.idx2frag = metadata['idx2frag']

        self.cell2idx = metadata['cell2idx']
        for idx, row in self.sig_info.iterrows():
            if row['cell_iname'] not in self.cell2idx.keys():
                self.sig_info.drop(index=idx, inplace=True)
        self.cell_lines = torch.from_numpy(np.array([self.cell2idx[cell_line] for cell_line in self.sig_info['cell_iname']], dtype=np.int64))

        self.max_len = metadata['max_len']
        self.vocab_size = metadata['vocab_size']
        self.padding_idx = metadata['padding_idx']
        
        # frags_tokens = []
        # for frags in self.sig_info['fragments']:
        #     frag_tokens = [self.frag2idx['[START]']]
        #     for frag in frags:
        #         frag_tokens.append(self.frag2idx[frag])
        #     frag_tokens.append(self.frag2idx['[END]'])
        #     for _ in range(self.max_len - len(frags)):
        #         frag_tokens.append(self.padding_idx)
        #     frags_tokens.append(frag_tokens)
        
        # self.frags_tokens = torch.from_numpy(np.array(frags_tokens, np.int64))

    def __getitem__(self, index):
        # return self.ge_emb[index], self.cell_lines[index], self.frags_tokens[index]
        return self.ge_emb[index], self.cell_lines[index]

    def __len__(self):
        return len(self.cell_lines)