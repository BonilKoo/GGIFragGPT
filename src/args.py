import argparse

def get_common_model_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset_name', type=str, default='experiment', help='Name of the dataset for saving results.')
    parser.add_argument('--out_path', type=str, default='./result', help='Path to save the results.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and testing.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--d_model', type=int, default=16, help='Dimension of the model.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads in the model.')
    parser.add_argument('--d_cell_line', type=int, default=4, help='Dimension of the cell line embeddings.')
    parser.add_argument('--PE_dropout', type=float, default=0.1, help='Dropout rate for positional encodings.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for the model.')
    parser.add_argument('--d_ff', type=int, default=64, help='Dimension of the feed-forward network in the model.')
    parser.add_argument('--act_func', type=str, default='relu', help='Activation function to use in the model.')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers in the model.')
    return parser

def get_common_data_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ge_emb', type=str, default='./data/extracted_geneformer_embs.pt', help='Path to the gene embeddings file.')
    parser.add_argument('--sig_data', type=str, default='./data/LINCS/processed_siginfo_beta_trt_cp.tsv', help='Path to the signature data file.')
    return parser

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parser for training and testing the molecule generation model."
    )
    subparsers = parser.add_subparsers(dest='command', help='Subcommands.')

    # 공통 parser 정의
    model_args = get_common_model_args()
    data_args = get_common_data_args()

    ########## train ##########
    subparser_train = subparsers.add_parser(
        'train', help='train', parents=[model_args, data_args]
    )
    subparser_train.add_argument('--data_path', type=str, default='./data', help='Path to the dataset directory.')
    subparser_train.add_argument('--frag_dict', type=str, default='./data/LINCS/fragment_dict.pkl', help='Path to the fragment dictionary file.')
    subparser_train.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of the dataset to use for validation.')
    subparser_train.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of the dataset to use for testing.')
    subparser_train.add_argument('--epochs', type=int, default=1000, help='Number of training epochs.')

    ########## test ##########
    subparser_test = subparsers.add_parser(
        'test', help='test', parents=[model_args]
    )

    ########## generate ##########
    subparser_generate = subparsers.add_parser(
        'generate', help='generate', parents=[model_args, data_args]
    )
    subparser_generate.add_argument('--n_mols', type=int, default=1, help='Number of molecules to generate.')
    subparser_generate.add_argument('--gen_file', type=str, default='generated.csv', help='Filename for saving generated molecules.')

    args = parser.parse_args()
    return args