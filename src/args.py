import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parser for training and testing the molecule generation model."
    )

    # fine-tuning, training, or get_attention
    parser.add_argument(
        '--ge_emb', type=str, default='../../data/LINCS/Geneformer/processed_level5_beta_trt_cp.pt',
        help="Path to the gene expression embedding file used for training."
    )

    # pre-training, training, or get_attention
    parser.add_argument(
        '--sig_data', type=str, default='../../data/LINCS/final/processed_siginfo_beta_trt_cp.tsv',
        help="Path to the signature data file (TSV format)."
    )

    # get_attention
    parser.add_argument(
        '--geneformer_dataset', type=str, default='../../data/LINCS/case/1_vorinostat/Geneformer_vorinostat.tsv',
        help=""
    )
    parser.add_argument(
        '--geneformer_token', type=str, default='../../data/LINCS/Geneformer/LINCS_lm_token_95M.csv',
        help=""
    )
    parser.add_argument(
        '--attn_out', type=str, default='../../result/attention/1_vorinostat.tsv',
        help=""
    )

    # fine-tuning
    parser.add_argument(
        '--frag_dict_pretrained', type=str, default='../../data/ChEMBL/fragment/fragment_dict.pkl',
        help=""
    )
    parser.add_argument(
        '--pretrained_ckpts', type=str, default='../../result/pretrain/ckpts_test/dim128_n6h8ff512_bs512_lr0.0001/best_model.ckpt',
        help=""
    )

    # pre-training or training
    parser.add_argument(
        '--data_path', type=str, default='../../data',
        help="Path to the root directory containing all data files."
    )
    parser.add_argument(
        '--frag_dict', type=str, default='../../data/LINCS/fragment/fragment_dict.pkl',
        help="Path to the fragment dictionary file (pickle format)."
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.1,
        help="Ratio of the dataset to be used for validation."
    )
    parser.add_argument(
        '--test_ratio', type=float, default=0.1,
        help="Ratio of the dataset to be used for testing."
    )
    parser.add_argument(
        '--epochs', type=int, default=1000,
        help="Number of training epochs."
    )

    # generate
    parser.add_argument(
        '--n_mols', type=int, default=1,
        help=""
    )
    parser.add_argument(
        '--gen_file', type=str, default='generated.csv',
        help=""
    )

    # pre-training, training, fine-tuing
    parser.add_argument(
        '--dataset_name', type=str, default='test',
        help="Name of the dataset to be used."
    )
    parser.add_argument(
        '--out_path', type=str, default='../../result/test',
        help="Output directory where results and checkpoints will be saved."
    )
    
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help="Device to run the training on (e.g., 'cuda:0' for GPU or 'cpu')."
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help="Batch size for training (for pretraining, a larger size such as 512 might be used)."
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001,
        help="Learning rate for the optimizer."
    )

    parser.add_argument(
        '--d_model', type=int, default=128,
        help="Dimension of the model's hidden representations."
    )
    parser.add_argument(
        '--n_heads', type=int, default=8,
        help="Number of attention heads in the transformer."
    )
    parser.add_argument(
        '--d_cell_line', type=int, default=4,
        help="Dimension of the cell line embedding."
    )
    parser.add_argument(
        '--PE_dropout', type=float, default=0.1,
        help="Dropout rate for the positional encoding module."
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help="General dropout rate for the model."
    )
    parser.add_argument(
        '--d_ff', type=int, default=512,
        help="Dimension of the feed-forward network within the transformer."
    )
    parser.add_argument(
        '--act_func', type=str, default='relu',
        help="Activation function to use (e.g., 'relu')."
    )
    parser.add_argument(
        '--n_layers', type=int, default=6,
        help="Number of transformer decoder layers."
    )
    
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args

def parse_args2():
    parser = argparse.ArgumentParser(
        description="Parser for training and testing the molecule generation model."
    )
    subparsers = parser.add_subparsers(dest='command', help='Subcommands.')

    ##########
    subparser_train = subparsers.add_parser('train', help='train')

    subparser_train.add_argument(
            '--ge_emb', type=str, default='./data/extracted_geneformer_embs.pt',
            help="Path to the gene expression embedding file used for training."
        )
    subparser_train.add_argument(
        '--sig_data', type=str, default='./data/LINCS/processed_siginfo_beta_trt_cp.tsv',
        help="Path to the signature data file (TSV format)."
    )
    subparser_train.add_argument(
        '--data_path', type=str, default='./data',
        help="Path to the root directory containing all data files."
    )
    subparser_train.add_argument(
        '--frag_dict', type=str, default='./data/LINCS/fragment_dict.pkl',
        help="Path to the fragment dictionary file (pickle format)."
    )
    subparser_train.add_argument(
        '--val_ratio', type=float, default=0.1,
        help="Ratio of the dataset to be used for validation."
    )
    subparser_train.add_argument(
        '--test_ratio', type=float, default=0.1,
        help="Ratio of the dataset to be used for testing."
    )
    subparser_train.add_argument(
        '--epochs', type=int, default=1000,
        help="Number of training epochs."
    )
    subparser_train.add_argument(
        '--dataset_name', type=str, default='experiment',
        help="Name of the dataset to be used."
    )
    subparser_train.add_argument(
        '--out_path', type=str, default='./result',
        help="Output directory where results and checkpoints will be saved."
    )
    subparser_train.add_argument(
        '--device', type=str, default='cuda:0',
        help="Device to run the training on (e.g., 'cuda:0' for GPU or 'cpu')."
    )
    subparser_train.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility."
    )
    subparser_train.add_argument(
        '--batch_size', type=int, default=128,
        help="Batch size for training (for pretraining, a larger size such as 512 might be used)."
    )
    subparser_train.add_argument(
        '--lr', type=float, default=0.0001,
        help="Learning rate for the optimizer."
    )
    subparser_train.add_argument(
        '--d_model', type=int, default=16,
        help="Dimension of the model's hidden representations."
    )
    subparser_train.add_argument(
        '--n_heads', type=int, default=8,
        help="Number of attention heads in the transformer."
    )
    subparser_train.add_argument(
        '--d_cell_line', type=int, default=4,
        help="Dimension of the cell line embedding."
    )
    subparser_train.add_argument(
        '--PE_dropout', type=float, default=0.1,
        help="Dropout rate for the positional encoding module."
    )
    subparser_train.add_argument(
        '--dropout', type=float, default=0.1,
        help="General dropout rate for the model."
    )
    subparser_train.add_argument(
        '--d_ff', type=int, default=64,
        help="Dimension of the feed-forward network within the transformer."
    )
    subparser_train.add_argument(
        '--act_func', type=str, default='relu',
        help="Activation function to use (e.g., 'relu')."
    )
    subparser_train.add_argument(
        '--n_layers', type=int, default=6,
        help="Number of transformer decoder layers."
    )
    ##########

    ##########
    subparser_test = subparsers.add_parser('test', help='test')

    subparser_test.add_argument(
        '--dataset_name', type=str, default='experiment',
        help="Name of the dataset to be used."
    )
    subparser_test.add_argument(
        '--out_path', type=str, default='./result',
        help="Output directory where results and checkpoints will be saved."
    )

    subparser_test.add_argument(
        '--device', type=str, default='cuda:0',
        help="Device to run the training on (e.g., 'cuda:0' for GPU or 'cpu')."
    )
    subparser_test.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility."
    )
    subparser_test.add_argument(
        '--batch_size', type=int, default=128,
        help="Batch size for training (for pretraining, a larger size such as 512 might be used)."
    )
    subparser_test.add_argument(
        '--lr', type=float, default=0.0001,
        help="Learning rate for the optimizer."
    )

    subparser_test.add_argument(
        '--d_model', type=int, default=16,
        help="Dimension of the model's hidden representations."
    )
    subparser_test.add_argument(
        '--n_heads', type=int, default=8,
        help="Number of attention heads in the transformer."
    )
    subparser_test.add_argument(
        '--d_cell_line', type=int, default=4,
        help="Dimension of the cell line embedding."
    )
    subparser_test.add_argument(
        '--PE_dropout', type=float, default=0.1,
        help="Dropout rate for the positional encoding module."
    )
    subparser_test.add_argument(
        '--dropout', type=float, default=0.1,
        help="General dropout rate for the model."
    )
    subparser_test.add_argument(
        '--d_ff', type=int, default=64,
        help="Dimension of the feed-forward network within the transformer."
    )
    subparser_test.add_argument(
        '--act_func', type=str, default='relu',
        help="Activation function to use (e.g., 'relu')."
    )
    subparser_test.add_argument(
        '--n_layers', type=int, default=6,
        help="Number of transformer decoder layers."
    )
    ##########

    ##########
    subparser_generate = subparsers.add_parser('generate', help='generate')

    subparser_generate.add_argument(
        '--n_mols', type=int, default=1,
        help="“"
    )
    subparser_generate.add_argument(
        '--gen_file', type=str, default='generated.csv',
        help=""
    )
    subparser_generate.add_argument(
        '--device', type=str, default='cuda:0',
        help="Device to run the training on (e.g., 'cuda:0' for GPU or 'cpu')."
    )
    subparser_generate.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility."
    )
    subparser_generate.add_argument(
        '--batch_size', type=int, default=128,
        help="Batch size for training (for pretraining, a larger size such as 512 might be used)."
    )
    subparser_generate.add_argument(
        '--lr', type=float, default=0.0001,
        help="Learning rate for the optimizer."
    )

    subparser_generate.add_argument(
        '--d_model', type=int, default=16,
        help="Dimension of the model's hidden representations."
    )
    subparser_generate.add_argument(
        '--n_heads', type=int, default=8,
        help="Number of attention heads in the transformer."
    )
    subparser_generate.add_argument(
        '--d_cell_line', type=int, default=4,
        help="Dimension of the cell line embedding."
    )
    subparser_generate.add_argument(
        '--PE_dropout', type=float, default=0.1,
        help="Dropout rate for the positional encoding module."
    )
    subparser_generate.add_argument(
        '--dropout', type=float, default=0.1,
        help="General dropout rate for the model."
    )
    subparser_generate.add_argument(
        '--d_ff', type=int, default=64,
        help="Dimension of the feed-forward network within the transformer."
    )
    subparser_generate.add_argument(
        '--act_func', type=str, default='relu',
        help="Activation function to use (e.g., 'relu')."
    )
    subparser_generate.add_argument(
        '--n_layers', type=int, default=6,
        help="Number of transformer decoder layers."
    )
    ##########
    
    args = parser.parse_args()
    # args, _ = parser.parse_known_args()
    return args