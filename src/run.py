from tqdm import tqdm
import torch
from torch.utils.data import random_split

# from args import parse_args
from args import parse_args2
from dataset import TrainDataset, TestDataset, GenerationDataset
from model import GGIFragGPT
from trainer import ModelTrainer, MoleculeGenerator
from utils import seed_everything, save_data, load_model
from evaluation import evaluate

def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    mydataset = TrainDataset(args.data_path, args.dataset_name, args.ge_emb, args.sig_data, args.frag_dict)
    dataset_size = len(mydataset)
    train_ratio = 1 - args.val_ratio - args.test_ratio
    train_dataset, val_dataset, test_dataset = random_split(mydataset, [train_ratio, args.val_ratio, args.test_ratio])
    args.ge_dim = train_dataset.dataset.ge_emb.shape[-1]
    save_data(mydataset, train_dataset.indices, val_dataset.indices, test_dataset.indices, args)

    model = GGIFragGPT(mydataset, args)

    trainer = ModelTrainer(model, train_dataset, val_dataset, test_dataset, device, args)
    trainer.train()

def test(args):
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
    result.to_csv(f'{args.out_path}/{args.dataset_name}/test.csv')

def generate(args):
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
    # args = parse_args()
    args = parse_args2()

    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    elif args.command == 'generate':
        generate(args)
    else:
        print(f"Unknown command: {args.command}. Please use 'train', 'test', or 'generate'.")
        exit(1)