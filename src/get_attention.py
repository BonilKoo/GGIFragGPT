import ast
import pandas as pd

import torch

from args import parse_args
from datasets import AttentionDataset
from evaluation import evaluate
from trainer import GetAttention
from utils import seed_everything, load_model

def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        attn_weights = module_out[1]
        self.outputs.append(attn_weights.detach().cpu())

    def clear(self):
        self.outputs.clear()

def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    attention_dataset = AttentionDataset(args.data_path, args.dataset_name, args.ge_emb, args.sig_data)
    args.ge_dim = attention_dataset.ge_emb.shape[-1]
    
    model = load_model(args, attention_dataset)

    save_output = SaveOutput()
    patch_attention(model.decoder.layers[-1].multihead_attn)
    hook_handle = model.decoder.layers[-1].multihead_attn.register_forward_hook(save_output)

    trainer = GetAttention(model, attention_dataset, device, args)
    trainer.val()
    # attn_weights = trainer.get_attention_weights().detach().cpu()

    attn_weights = torch.cat(save_output.outputs, dim=0)
    save_output = pd.DataFrame(attn_weights.mean(dim=1).sum(dim=1))

    sig_data = pd.read_table(attention_dataset.sig_file)

    geneformer_info = pd.read_csv(args.geneformer_dataset, index_col=0).reset_index(drop=True)
    geneformer_info['input_ids'] = [ast.literal_eval(i) for i in geneformer_info['input_ids']]

    df_LINCS_lm_token = pd.read_csv(args.geneformer_token)
    dict_LINCS_lm_token = {row['token']: row['gene_name'] for idx, row in df_LINCS_lm_token.iterrows()}

    sig_data['attn_gene_rank'] = None
    sig_data['attn_gene_rank'] = sig_data['attn_gene_rank'].astype(object)
    sample_gene_score = dict()
    for idx, row in geneformer_info.iterrows():
        sorted_genes = row['input_ids'][1:]
        sorted_genes = [dict_LINCS_lm_token[i] for i in sorted_genes]
        gene_scores = save_output.loc[idx]
        gene_scores.index = sorted_genes
        gene_scores.sort_values(ascending=False, inplace=True)
        sig_data.at[idx, 'attn_gene_rank'] = gene_scores.index.to_list()

    output = f''
    sig_data.to_csv(args.attn_out, sep='\t', index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)