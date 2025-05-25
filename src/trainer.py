import pathlib
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import LabelSmoothing, subsequent_mask
from utils import replace_dummy_atoms_in_smiles

class ModelTrainer():
    def __init__(self, model, train_dataset, val_dataset, test_dataset, device, args, records=None):
        self.device = device
        self.model = model.to(self.device)
        self.args = args

        if records is None:
            self.records = {
                'train_losses': [],
                'train_record': [],
                'val_losses': [],
                'val_record': [],
                'best_ckpt': None
            }
        else:
            self.records = records

        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        self.criterion = LabelSmoothing(size=train_dataset.dataset.vocab_size,
                                        padding_idx=train_dataset.dataset.padding_idx,
                                        smoothing=0.1).to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=0.7, patience=4, min_lr=1e-6)

        self.folder = pathlib.Path(f'{self.args.out_path}/ckpts_{self.args.dataset_name}') / \
                      f'dim{self.args.d_model}_n{self.args.n_layers}h{self.args.n_heads}ff{self.args.d_ff}_' \
                      f'bs{self.args.batch_size}_lr{self.args.lr}'
        self.folder.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.start = time.time()

        self.log(f'train set num: {len(train_dataset)}    val set num: {len(val_dataset)}    test set num: {len(test_dataset)}')
        total_params = sum([p.nelement() for p in self.model.parameters()])
        self.log(f'total parameters: {total_params}')
        self.log(msgs=str(model).split('\n'), show=False)

    def save_best_model(self, epoch, val_loss):
        checkpoint_path = self.folder / 'best_model.ckpt'
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'records': self.records,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        self.records['best_ckpt'] = str(checkpoint_path)
        self.log(f'Best model saved at epoch {epoch} with val_loss {val_loss:.5f}')

    def save_loss_records(self):
        recorders_folder = self.folder / 'recorders'
        recorders_folder.mkdir(parents=True, exist_ok=True)
        train_record = pd.DataFrame(self.records['train_record'], columns=['epoch', 'train_loss', 'lr'])
        val_record = pd.DataFrame(self.records['val_record'], columns=['epoch', 'val_loss', 'lr'])
        ret = pd.DataFrame({
            'epoch': train_record['epoch'],
            'train_loss': train_record['train_loss'],
            'val_loss': val_record['val_loss'],
            'train_lr': train_record['lr'],
            'val_lr': val_record['lr'],
        })
        csv_path = recorders_folder / f'record_dim{self.args.d_model}_n{self.args.n_layers}h{self.args.n_heads}ff' \
                                      f'{self.args.d_ff}_bs{self.args.batch_size}_lr{self.args.lr}.csv'
        ret.to_csv(csv_path, index=False)
        return ret

    def train_iterations(self):
        self.model.train()
        losses = []
        for data in tqdm(self.train_dataloader):
            ge, cell_lines, frags_tokens = data
            ge = ge.to(self.device)
            cell_lines = cell_lines.to(self.device)
            frags_tokens = frags_tokens.to(self.device)

            out_pred = self.model(ge, cell_lines, frags_tokens)

            loss = self.criterion(out_pred, frags_tokens[:, 1:]) / out_pred.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        train_loss = np.mean(losses)

        return train_loss

    def val_iterations(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in tqdm(self.val_dataloader):
                ge, cell_lines, frags_tokens = data
                ge = ge.to(self.device)
                cell_lines = cell_lines.to(self.device)
                frags_tokens = frags_tokens.to(self.device)

                out_pred = self.model(ge, cell_lines, frags_tokens)

                loss = self.criterion(out_pred, frags_tokens[:, 1:]) / out_pred.size(0)

                losses.append(loss.item())

        val_loss = np.mean(losses)

        return val_loss

    def train(self):
        self.log('Training start')
        early_stop_cnt = 0
        patience_limit = 10
        
        for epoch in range(1, self.args.epochs+1):
            train_loss = self.train_iterations()
            val_loss = self.val_iterations()

            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']

            self.log(f'Epoch:{epoch} train_loss:{train_loss:.5f} lr_cur:{lr_cur:.5f}', with_time=True)
            self.log(f'Epoch:{epoch} val_loss:{val_loss:.5f} lr_cur:{lr_cur:.5f}', with_time=True)

            self.records['train_losses'].append(train_loss)
            self.records['val_losses'].append(val_loss)
            self.records['train_record'].append([epoch, train_loss, lr_cur])
            self.records['val_record'].append([epoch, val_loss, lr_cur])

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_best_model(epoch, val_loss)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
                if early_stop_cnt >= patience_limit:
                    self.log(f'Early stopping at epoch {epoch}')
                    break

        self.save_loss_records

    def log(self, msg=None, msgs=None, with_time=False, show=True):
        if with_time and msg:
            elapsed_secs = time.time() - self.start
            hrs = elapsed_secs / 3600.
            mins = elapsed_secs / 60.
            msg = msg + f' time elapsed {hrs:2f} hrs ({mins:.1f} mins)'

        log_folder = pathlib.Path(f'{self.args.out_path}/ckpts_{self.args.dataset_name}/log')
        log_folder.mkdir(parents=True, exist_ok=True)
        log_file = log_folder / 'log.txt'
        
        with open(log_file, 'a+') as f:
            if msgs:
                separator = '#' * 80 + '\n'
                f.write(separator)
                for m in msgs:
                    f.write(m + '\n')
                    if show:
                        print(m)
            if msg:
                f.write(msg + '\n')
                if show:
                    print(msg)

class MoleculeGenerator():
    def __init__(self, model, dataset, device, args):
        self.device = device
        self.model = model.to(self.device)
        self.args = args
        self.max_len = dataset.max_len
        self.idx2frag = dataset.idx2frag
        self.frag2idx = dataset.frag2idx
        self.start_idx = self.frag2idx['[START]']
        
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    def generate(self):
        self.model.eval()
        generated = []
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                ge, cell_lines = data
                ge = ge.to(self.device)
                cell_lines = cell_lines.to(self.device)
                
                out_gen = self.model.generate(ge, cell_lines, self.max_len)
                
                generated.extend(out_gen)
                
        generated = replace_dummy_atoms_in_smiles(generated)

        return generated

class PreTrainer():
    def __init__(self, model, train_dataset, val_dataset, test_dataset, device, args, records=None):
        self.device = device
        self.model = model.to(self.device)
        self.args = args

        if records is None:
            self.records = {
                'train_losses': [],
                'train_record': [],
                'val_losses': [],
                'val_record': [],
                'best_ckpt': None
            }
        else:
            self.records = records

        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        self.criterion = LabelSmoothing(size=train_dataset.dataset.vocab_size,
                                        padding_idx=train_dataset.dataset.padding_idx,
                                        smoothing=0.1).to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=args.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=0.7, patience=4, min_lr=1e-6)

        self.folder = pathlib.Path(f'{self.args.out_path}/ckpts_{self.args.dataset_name}') / \
                      f'dim{self.args.d_model}_n{self.args.n_layers}h{self.args.n_heads}ff{self.args.d_ff}_' \
                      f'bs{self.args.batch_size}_lr{self.args.lr}'
        self.folder.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.start = time.time()

        self.log(f'train set num: {len(train_dataset)}    val set num: {len(val_dataset)}    test set num: {len(test_dataset)}')
        total_params = sum([p.nelement() for p in self.model.parameters()])
        self.log(f'total parameters: {total_params}')
        self.log(msgs=str(model).split('\n'), show=False)

    def save_best_model(self, epoch, val_loss):
        checkpoint_path = self.folder / 'best_model.ckpt'
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'records': self.records,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        self.records['best_ckpt'] = str(checkpoint_path)
        self.log(f'Best model saved at epoch {epoch} with val_loss {val_loss:.5f}')

    def save_loss_records(self):
        recorders_folder = self.folder / 'recorders'
        recorders_folder.mkdir(parents=True, exist_ok=True)
        train_record = pd.DataFrame(self.records['train_record'], columns=['epoch', 'train_loss', 'lr'])
        val_record = pd.DataFrame(self.records['val_record'], columns=['epoch', 'val_loss', 'lr'])
        ret = pd.DataFrame({
            'epoch': train_record['epoch'],
            'train_loss': train_record['train_loss'],
            'val_loss': val_record['val_loss'],
            'train_lr': train_record['lr'],
            'val_lr': val_record['lr'],
        })
        csv_path = recorders_folder / f'record_dim{self.args.d_model}_n{self.args.n_layers}h{self.args.n_heads}ff' \
                                      f'{self.args.d_ff}_bs{self.args.batch_size}_lr{self.args.lr}.csv'
        ret.to_csv(csv_path, index=False)
        return ret

    def train_iterations(self):
        self.model.train()
        losses = []
        for frags_tokens in tqdm(self.train_dataloader):
            frags_tokens = frags_tokens.to(self.device)
            
            out_pred = self.model(frags_tokens)

            loss = self.criterion(out_pred, frags_tokens[:, 1:]) / out_pred.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        train_loss = np.mean(losses)

        return train_loss

    def val_iterations(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for frags_tokens in tqdm(self.val_dataloader):
                frags_tokens = frags_tokens.to(self.device)
                
                out_pred = self.model(frags_tokens)

                loss = self.criterion(out_pred, frags_tokens[:, 1:]) / out_pred.size(0)

                losses.append(loss.item())

        val_loss = np.mean(losses)

        return val_loss

    def train(self):
        self.log('Training start')
        early_stop_cnt = 0
        patience_limit = 10
        
        for epoch in range(1, self.args.epochs+1):
            train_loss = self.train_iterations()
            val_loss = self.val_iterations()

            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']

            self.log(f'Epoch:{epoch} train_loss:{train_loss:.5f} lr_cur:{lr_cur:.5f}', with_time=True)
            self.log(f'Epoch:{epoch} val_loss:{val_loss:.5f} lr_cur:{lr_cur:.5f}', with_time=True)

            self.records['train_losses'].append(train_loss)
            self.records['val_losses'].append(val_loss)
            self.records['train_record'].append([epoch, train_loss, lr_cur])
            self.records['val_record'].append([epoch, val_loss, lr_cur])

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_best_model(epoch, val_loss)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
                if early_stop_cnt >= patience_limit:
                    self.log(f'Early stopping at epoch {epoch}')
                    break

        self.save_loss_records()

    def log(self, msg=None, msgs=None, with_time=False, show=True):
        if with_time and msg:
            elapsed_secs = time.time() - self.start
            hrs = elapsed_secs / 3600.
            mins = elapsed_secs / 60.
            msg = msg + f' time elapsed {hrs:2f} hrs ({mins:.1f} mins)'

        log_folder = pathlib.Path(f'{self.args.out_path}/ckpts_{self.args.dataset_name}/log')
        log_folder.mkdir(parents=True, exist_ok=True)
        log_file = log_folder / 'log.txt'
        
        with open(log_file, 'a+') as f:
            if msgs:
                separator = '#' * 80 + '\n'
                f.write(separator)
                for m in msgs:
                    f.write(m + '\n')
                    if show:
                        print(m)
            if msg:
                f.write(msg + '\n')
                if show:
                    print(msg)

class GetAttention():
    def __init__(self, model, dataset, device, args):
        self.device = device
        self.model = model.to(self.device)
        self.args = args

        self.val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    def val_iterations(self):
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.val_dataloader):
                ge, cell_lines, frags_tokens = data
                ge = ge.to(self.device)
                cell_lines = cell_lines.to(self.device)
                frags_tokens = frags_tokens.to(self.device)

                self.model(ge, cell_lines, frags_tokens)

    def val(self):
        self.val_iterations()

# class GetAttention():
#     def __init__(self, model, dataset, device, args):
#         self.device = device
#         self.model = model.to(self.device)
#         self.args = args
#         self.pad_idx = dataset.frag2idx['[PAD]']

#         self.val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
#         self.attn_weights_all = []

#     def val_iterations(self):
#         self.model.eval()
#         with torch.no_grad():
#             for data in tqdm(self.val_dataloader):
#                 ge, cell_lines, frags_tokens = data
#                 ge = ge.to(self.device)
#                 cell_lines = cell_lines.to(self.device)
#                 frags_tokens = frags_tokens.to(self.device)

#                 # forward
#                 ge_proj = self.model.gene_proj_layer(ge)
#                 cell_lines_emb = self.model.cell_line_embedding(cell_lines)
#                 cell_lines_emb = cell_lines_emb.unsqueeze(1).expand(-1, ge_proj.size(1), -1)
#                 ge_cell = torch.cat([ge_proj, cell_lines_emb], dim=-1)
#                 memory = self.model.gene_cell_line_proj_layer(ge_cell)

#                 tgt = frags_tokens[:, :-1]
#                 tgt_emb = self.model.frag_embedding(tgt)
#                 tgt_emb = self.model.position(tgt_emb)

#                 tgt_mask = subsequent_mask(tgt_emb.size(1), device=ge.device)
#                 tgt_key_padding_mask = (tgt == self.pad_idx)

#                 # extract attention weights directly
#                 decoder_layer = self.model.decoder.layers[-1]
#                 _, attn_weights = decoder_layer.multihead_attn(
#                     tgt_emb, memory, memory,
#                     # attn_mask=tgt_mask,
#                     key_padding_mask=None,
#                     need_weights=True,
#                     average_attn_weights=False
#                 ) # (batch, n_heads, tgt_len, src_len)

#                 # mask padding tokens (exclude padded position from tgt side)

#                 self.model(ge, cell_lines, frags_tokens)

#                 attn_weights = attn_weights.masked_fill(tgt_key_padding_mask.unsqueeze(1).unsqueeze(-1), 0.0)
#                 self.attn_weights_all.append(attn_weights)

#     def val(self):
#         self.val_iterations()
    
#     def get_attention_weights(self):
#         return torch.cat(self.attn_weights_all, dim=0) # (total_batch, n_heads, tgt_len, src_len)