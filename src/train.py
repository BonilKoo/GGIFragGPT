import torch
from torch.utils.data import random_split

from args import parse_args
from dataset import TrainDataset
from model import FragmentTransformer
from trainer import ModelTrainer
from utils import seed_everything, save_data

def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    mydataset = TrainDataset(args.data_path, args.dataset_name, args.ge_emb, args.sig_data, args.frag_dict)
    dataset_size = len(mydataset)
    train_ratio = 1 - args.val_ratio - args.test_ratio
    train_dataset, val_dataset, test_dataset = random_split(mydataset, [train_ratio, args.val_ratio, args.test_ratio])
    args.ge_dim = train_dataset.dataset.ge_emb.shape[-1]
    save_data(mydataset, train_dataset.indices, val_dataset.indices, test_dataset.indices, args)

    model = FragmentTransformer(mydataset, args)

    trainer = ModelTrainer(model, train_dataset, val_dataset, test_dataset, device, args)
    trainer.train()

if __name__ == '__main__':
    # args = parse_args()
    args = parse_args2()
    main(args)