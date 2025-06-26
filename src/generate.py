from tqdm import tqdm

import torch
from torch.utils.data import random_split

from args import parse_args
from dataset import GenerationDataset
from evaluation import evaluate
from trainer import MoleculeGenerator
from utils import seed_everything, load_model

def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = GenerationDataset(args.data_path, args.dataset_name, args.ge_emb, args.sig_data)
    args.ge_dim = dataset.ge_emb.shape[-1]
    
    model = load_model(args, dataset)

    generator = MoleculeGenerator(model, dataset, device, args)
    generated = []
    for _ in tqdm(range(args.n_mols)):
        generated.extend(generator.generate())

    result = dataset.sig_info
    result = result.loc[result.index.to_list() * args.n_mols]
    result['generated'] = generated
    result.to_csv(f'{args.out_path}/{args.dataset_name}/{args.gen_file}')

if __name__ == '__main__':
    args = parse_args()
    main(args)