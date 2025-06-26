import argparse

from cmapPy.pandasGEXpress.parse import parse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--target',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--sig_data',
        type=str,
        default='./data/LINCS/siginfo_beta_trt_sh.tsv',
    )

    parser.add_argument(
        '--ge',
        type=str,
        default='./data/LINCS/level5_beta_trt_sh.tsv',
    )

    args = parser.parse_args()
    return args

def load_data(args):
    sig_info = pd.read_table(args.sig_data)
    sig_info_target = sig_info[sig_info['cmap_name'] == args.target]
    if len(sig_info_target) == 0:
        raise ValueError(f"No signatures found for target {args.target} in {args.sig_data}")
    
    level5_sh = pd.read_table(args.ge, index_col=0)
    level5_sh_target = level5_sh.loc[sig_info_target['sig_id']]

    return sig_info_target, level5_sh_target

def save_data(sig_info_target, level5_sh_target, target):
    sig_info_target.to_csv(f'./data/generation/siginfo_beta_trt_sh_{target}.tsv', sep='\t', index=False)
    level5_sh_target.to_csv(f'./data/generation/level5_beta_trt_sh_{target}.tsv', sep='\t')

def main(args):
    sig_info_target, level5_sh_target = load_data(args)
    save_data(sig_info_target, level5_sh_target, args.target)

if __name__ == '__main__':
    args = parse_args()
    main(args)