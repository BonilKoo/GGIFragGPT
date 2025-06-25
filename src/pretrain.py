import pickle

import torch
from torch.utils.data import random_split

from args import parse_args
from dataset import ChEMBLDataset
from model import FragmentGPT
from trainer import PreTrainer
from utils import seed_everything, save_data

def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    chembl_dataset = ChEMBLDataset(args.data_path, args.dataset_name, args.sig_data, args.frag_dict)
    dataset_size = len(chembl_dataset)
    train_ratio = 1 - args.val_ratio - args.test_ratio
    train_dataset, val_dataset, test_dataset = random_split(chembl_dataset, [train_ratio, args.val_ratio, args.test_ratio])
    
    model = FragmentGPT(chembl_dataset, args)

    trainer = PreTrainer(model, train_dataset, val_dataset, test_dataset, device, args)
    trainer.train()

if __name__ == '__main__':
    args = parse_args()
    main(args)