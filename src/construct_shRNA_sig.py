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

    parser.add_argument()

    args = parser.parse_args()
    return args

def load_data(args):
    sig_info = pd.read_table(args.sig_data)
    sig_info_target = sig_info[(sig_info['cmap_name'] == args.target) & (sig_info['sig_id'].str.contains(args.target))]
    if len(sig_info_target) == 0:
        raise ValueError(f"No signatures found for target {args.target} in {args.sig_data}")
    
    level5_sh = pd.read_table(args.ge, index_col=0)
    level5_sh_target = level5_sh[sig_info_target['sig_id']]

    return sig_info_target, level5_sh_target

def main(args):
    sig_info_target, level5_sh_target = load_data(args)

    

if __name__ == '__main__':
    args = parse_args()
    main(args)