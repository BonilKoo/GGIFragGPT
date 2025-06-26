import torch
from torch.utils.data import random_split

from args import parse_args
from dataset import TestDataset
from evaluation import evaluate
from trainer import MoleculeGenerator
from utils import seed_everything, load_model

def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    test_dataset = TestDataset(args.out_path, args.dataset_name)
    args.ge_dim = test_dataset.ge_emb.shape[-1]
    
    model = load_model(args, test_dataset)

    generator = MoleculeGenerator(model, test_dataset, device, args)
    generated = generator.generate()
    metrics = evaluate(generated, test_dataset.sig_info['canonical_smiles'], device)

    result = test_dataset.sig_info
    result['generated'] = generated
    result.to_csv(f'{args.out_path}/{args.dataset_name}/generated.csv')

if __name__ == '__main__':
    args = parse_args()
    main(args)