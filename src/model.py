import math
import pickle
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from chem_utils import ALLOWED_MAPPING, merge_fragment_smiles

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super(Embedding, self).__init__()
        self.LUT = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.LUT(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=13):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        # x: (bs, seq_len, vocab)
        # target: (bs, seq_len)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 2))

        true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)

        true_dist.masked_fill_((target.data == self.padding_idx).unsqueeze(-1), 0)

        return self.criterion(x, true_dist)

def load_pretrained_weights(model, args):
    """
    Function to update the model's frag_embedding and other parameters with pre-trained weights.
    
    Args:
        model: The fine-tuning target model (e.g., FragmentTransformer)
        pretrained_state_dict: The state_dict from the pre-trained model
        pretrained_frag2idx: The token-to-index dictionary used in the pre-trained model
    """
    with open(args.frag_dict_pretrained, 'rb') as f:
        pretrained_frag2idx = pickle.load(f)
    pretrained_state_dict = torch.load(args.pretrained_ckpts, weights_only=False)['model_state_dict']
    # 1. Update frag_embedding (handling token mapping)
    pretrained_embedding = pretrained_state_dict['frag_embedding.LUT.weight']
    embedding_dim = pretrained_embedding.shape[1]

    # Initialize a new embedding matrix with the model's total vocabulary size
    new_embedding_matrix = torch.randn(model.frag_embedding.LUT.weight.shape[0], embedding_dim)
    
    # For each token in the fine-tuning vocabulary that exists in the pre-trained vocabulary,
    # copy the corresponding embedding from the pre-trained model.
    for token, finetune_idx in model.frag2idx.items():
        if token in pretrained_frag2idx:
            pretrain_idx = pretrained_frag2idx[token]
            new_embedding_matrix[finetune_idx] = pretrained_embedding[pretrain_idx]
    
    # Copy the new embedding matrix into the model's frag_embedding layer
    model.frag_embedding.LUT.weight.data.copy_(new_embedding_matrix)
    
    # 2. Update the rest of the parameters (only update those with matching keys and shapes)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_state_dict.items()
        if k != 'frag_embedding.LUT.weight' and k in model_dict and v.size() == model_dict[k].size()
    }
    
    # Update the current model's state_dict with the pre-trained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    return model

class FragmentTransformer(nn.Module):
    def __init__(self, dataset, args):
        super(FragmentTransformer, self).__init__()
        self.params = args
        self.d_model = args.d_model
        self.frag2idx = dataset.frag2idx
        self.idx2frag = dataset.idx2frag
        self.pad_idx = dataset.frag2idx['[PAD]']
        self.n_heads = args.n_heads
        self.d_cell_line = args.d_cell_line

        self.gene_proj_layer = nn.Linear(args.ge_dim, self.d_model)
        self.cell_line_embedding = Embedding(len(dataset.cell2idx), self.d_cell_line)
        self.gene_cell_line_proj_layer = nn.Linear(self.d_model+self.d_cell_line, self.d_model)

        self.frag_embedding = Embedding(dataset.vocab_size, self.d_model, padding_idx=self.pad_idx)
        self.position = PositionalEncoding(self.d_model, args.PE_dropout, max_len=dataset.max_len)
                        
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=args.d_ff,
                                                   batch_first=True,
                                                   dropout=args.dropout,
                                                   activation=args.act_func)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.n_layers)
        self.generator = Generator(self.d_model, dataset.vocab_size)

    def forward(self, ge, cell_lines, fragments):
        # ge (bs, 978, 512)
        # cell_lines (bs)
        # fragments (bs, 13)
        ge = self.gene_proj_layer(ge) # (bs, 978, 64)
        cell_lines = self.cell_line_embedding(cell_lines) # (bs, 4)
        cell_lines = cell_lines.unsqueeze(1).expand(-1, ge.size()[1], -1) # (bs, 978, 4)
        
        ge_cell_lines = torch.cat([ge, cell_lines], dim=-1) # (bs, 978, 64+4)
        ge_cell_lines = self.gene_cell_line_proj_layer(ge_cell_lines) # (bs, 978, 64)

        fragments_emb = self.frag_embedding(fragments) # (bs, 13, 64)
        fragments_pos = self.position(fragments_emb) # (bs, 13, 64)

        tgt_seq_len = fragments.size(1) - 1 # 12

        tgt_mask = subsequent_mask(tgt_seq_len, fragments.device) # (12, 12)

        tgt_key_padding_mask = (fragments[:, :-1] == self.pad_idx) # (bs, 12)

        fragments_pred = self.decoder(
            fragments_pos[:, :-1],
            ge_cell_lines,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None
        )
        fragments_p = self.generator(fragments_pred)

        return fragments_p

    def generate(self, ge, cell_lines, max_len, top_p=0.9):
        # ge (bs, 978, 512)
        # cell_lines (bs, 13)
        batch_size = ge.size()[0]
        
        ge = self.gene_proj_layer(ge)
        cell_lines_emb = self.cell_line_embedding(cell_lines)
        cell_lines_emb = cell_lines_emb.unsqueeze(1).expand(-1, ge.size()[1], -1)

        ge_cell_lines = torch.cat([ge, cell_lines_emb], dim=-1)
        ge_cell_lines = self.gene_cell_line_proj_layer(ge_cell_lines)

        generated_idx_seq = torch.ones(batch_size, 1).fill_(self.frag2idx['[START]']).type_as(cell_lines.data)

        current_smiles = None
        finished = torch.zeros(batch_size, dtype=torch.bool)

        for t in range(max_len-1):
            fragments_emb = self.frag_embedding(generated_idx_seq) # (bs, len, d_model)
            fragments_pos = self.position(fragments_emb) # (bs, len, d_model)

            fragments_pred = self.decoder(
                fragments_pos,
                ge_cell_lines,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
            )
            fragments_logits = self.generator(fragments_pred[:, -1, :]) # (bs, vocab_size)
            pred_tokens = top_p_sampling(fragments_logits, current_smiles, self.frag2idx, top_p) # (bs, 1)
            
            generated_idx_seq = torch.cat([generated_idx_seq, pred_tokens], dim=1) # (bs, len)

            END_sample_idx = (pred_tokens.view(-1) == self.frag2idx['[END]']).nonzero()
            for idx in range(batch_size):
                if finished[idx] == False and idx in END_sample_idx:
                    finished[idx] = True
            if finished.sum() == batch_size:
                return current_smiles
            
            current_fragments = [self.idx2frag[idx] for idx in pred_tokens.reshape(-1).detach().cpu().tolist()]
            if t == 0:
                current_smiles = current_fragments
            else:
                current_smiles = merge_fragment_smiles(current_smiles, current_fragments, finished.tolist())
        return current_smiles

def subsequent_mask(size, device='cpu'):
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

def top_p_sampling(logits, current_smiles, frag2idx, top_p=0.0, filter_value=-float('Inf')):
    # Sample only from connectable fragments
    assert logits.dim() == 2

    if current_smiles is None:
        logits[:, [ frag2idx['[PAD]'], frag2idx['[UNK]'], frag2idx['[START]'] ] ] = filter_value
    else:
        # For each SMILES, mask logits for candidate tokens.
        for idx, smiles in enumerate(current_smiles):
            # Extract open attachment labels from the current SMILES.
            # For example, from "[5*]N[5*]" extract {'5'}
            current_labels = set(re.findall(r'\[([0-9]+)\*\]', smiles))
            
            # For each attachment label, collect all candidate labels that can be connected.
            allowed_labels = set()
            for lab in current_labels:
                allowed_labels.update(ALLOWED_MAPPING.get(lab, set()))
            
            # For each token in the vocabulary, if the token is a special token or 
            # the attachment labels extracted from the token do not overlap with allowed_labels at all, mask the logits.
            for token, token_idx in frag2idx.items():
                if token in ['[PAD]', '[UNK]', '[START]']:
                    logits[idx, token_idx] = filter_value
                elif token == '[END]':
                    continue
                else:
                    # Extract attachment labels from the candidate token.
                    token_labels = set(re.findall(r'\[([0-9]+)\*\]', token))
                    # If the token has no attachment labels or there is no overlap with the current allowed_labels, mask it.
                    if not token_labels or token_labels.isdisjoint(allowed_labels):
                        logits[idx, token_idx] = filter_value            
    
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_idx_to_remove = cumulative_probs > top_p
    
    sorted_idx_to_remove[:, 1:] = sorted_idx_to_remove[:, :-1].clone()
    sorted_idx_to_remove[:, 0] = 0
    
    sorted_logits[sorted_idx_to_remove] = filter_value
    
    logits = torch.gather(sorted_logits, dim=1, index=sorted_idx.argsort(dim=-1))

    pred_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    
    return pred_token

class FragmentGPT(nn.Module):
    def __init__(self, dataset, args):
        super(FragmentGPT, self).__init__()
        self.params = args
        self.d_model = args.d_model
        self.frag2idx = dataset.frag2idx
        self.idx2frag = dataset.idx2frag
        self.pad_idx = dataset.frag2idx['[PAD]']
        self.n_heads = args.n_heads

        self.frag_embedding = Embedding(dataset.vocab_size, self.d_model, padding_idx=self.pad_idx)
        self.position = PositionalEncoding(self.d_model, args.PE_dropout, max_len=dataset.max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=args.d_ff,
                                                   batch_first=True,
                                                   dropout=args.dropout,
                                                   activation=args.act_func)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.n_layers)
        self.generator = Generator(self.d_model, dataset.vocab_size)

    def forward(self, fragments):
        # ge (bs, 978, 512)
        # cell_lines (bs)
        # fragments (bs, 13)

        fragments_emb = self.frag_embedding(fragments) # (bs, 13, 64)
        fragments_pos = self.position(fragments_emb) # (bs, 13, 64)

        tgt_seq_len = fragments.size(1) - 1 # 12

        tgt_mask = subsequent_mask(tgt_seq_len, fragments.device) # (12, 12)

        tgt_key_padding_mask = (fragments[:, :-1] == self.pad_idx) # (bs, 12)

        fragments_pred = self.decoder(
            fragments_pos[:, :-1],
            fragments_pos[:, :-1],
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,    
            memory_key_padding_mask=tgt_key_padding_mask
        )
        fragments_p = self.generator(fragments_pred)

        return fragments_p