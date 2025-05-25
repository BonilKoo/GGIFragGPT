import os
import pathlib
import random
import numpy as np
import torch
import pickle
import pandas as pd

from rdkit import Chem

from model import FragmentTransformer

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministric = True
        torch.backends.cudnn.benchmark = False

def save_data(dataset, train_idx, val_idx, test_idx, args):
    save_path = f'{args.out_path}/ckpts_{args.dataset_name}/dataset'
    os.makedirs(save_path, exist_ok=True)

    with open(f'{save_path}/split_indices.pkl', 'wb') as f:
        pickle.dump({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }, f)
    
    with open(f'{save_path}/metadata.pkl', 'wb') as f:
        pickle.dump({
            'cell2idx': dataset.cell2idx,
            'frag2idx': dataset.frag2idx,
            'idx2frag': dataset.idx2frag,
            'max_len': dataset.max_len,
            'vocab_size': dataset.vocab_size,
            'padding_idx': dataset.padding_idx
        }, f)

    sig_info = pd.read_table(dataset.sig_file, index_col=0)

    # torch.save({
    #     'ge_emb': dataset.ge_emb[train_idx].type(torch.float16),
    #     'cell_lines': dataset.cell_lines[train_idx],
    #     'frags_tokens': dataset.frags_tokens[train_idx],
    #     'sig_info': sig_info.iloc[train_idx]
    # }, f'{save_path}/train_data.pt')

    # torch.save({
    #     'ge_emb': dataset.ge_emb[val_idx].type(torch.float16),
    #     'cell_lines': dataset.cell_lines[val_idx],
    #     'frags_tokens': dataset.frags_tokens[val_idx],
    #     'sig_info': sig_info.iloc[val_idx]
    # }, f'{save_path}/val_data.pt')

    torch.save({
        'ge_emb': dataset.ge_emb[test_idx].type(torch.float16),
        'cell_lines': dataset.cell_lines[test_idx],
        'frags_tokens': dataset.frags_tokens[test_idx],
        'sig_info': sig_info.iloc[test_idx]
    }, f'{save_path}/test_data.pt')

def load_model(args, dataset):
    dir_model = f'dim{args.d_model}_n{args.n_layers}h{args.n_heads}ff{args.d_ff}_bs{args.batch_size}_lr{args.lr}'
    path_model = f'{args.out_path}/ckpts_{args.dataset_name}/{dir_model}/best_model.ckpt'
    
    model = FragmentTransformer(dataset, args)
    
    ckpt = torch.load(path_model, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    return model

def replace_dummy_atoms(mol):
    """
    Processes dummy atoms (atomic number 0) in the input Mol object.
    - For dummy atoms that participate only in single bonds, replace them with hydrogen (H, atomic number 1)
      and remove their isotope, 'brics' property, and atom mapping number.
    - For dummy atoms involved in double bonds, remove the bond between the dummy atom and the heavy atom,
      then delete the dummy atom.
    Afterwards, recalculate the implicit hydrogens of the heavy atoms by performing Chem.AddHs followed by Chem.RemoveHs.
    """
    rw_mol = Chem.RWMol(mol)
    # First, collect the indices of all dummy atoms (atomic number 0) in descending order.
    dummy_indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in sorted(dummy_indices, reverse=True):
        # Check index validity (skip if the atom has already been deleted).
        if idx >= rw_mol.GetNumAtoms():
            continue
        atom = rw_mol.GetAtomWithIdx(idx)
        bonds = list(atom.GetBonds())
        # Check if the dummy atom is involved in any double bond.
        if any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds):
            # For double bonds: find the heavy atom (atomic number != 0) connected to the dummy atom.
            heavy_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 0]
            if len(heavy_neighbors) == 1:
                heavy_atom = heavy_neighbors[0]
                heavy_idx = heavy_atom.GetIdx()
                # Remove the bond between the heavy atom and the dummy atom.
                if rw_mol.GetBondBetweenAtoms(idx, heavy_idx):
                    rw_mol.RemoveBond(idx, heavy_idx)
                # Delete the dummy atom.
                rw_mol.RemoveAtom(idx)
            else:
                print(f"Warning: The number of heavy neighbors for dummy atom idx {idx} is not 1.")
                continue
        else:
            # If only single bonds exist: directly replace the dummy atom with hydrogen.
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            if atom.HasProp('brics'):
                atom.ClearProp('brics')
            atom.SetAtomMapNum(0)
    new_mol = rw_mol.GetMol()
    # To recalculate the implicit hydrogens of heavy atoms according to RDKit rules,
    # perform AddHs followed by RemoveHs.
    new_mol = Chem.AddHs(new_mol)
    Chem.SanitizeMol(new_mol)
    new_mol = Chem.RemoveHs(new_mol)
    return new_mol

def replace_dummy_atoms_in_smiles(smi_list):
    """
    Takes a list of SMILES strings as input, applies the replace_dummy_atoms function to each molecule,
    and returns the final list of SMILES strings.
    """
    new_smiles = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            new_smiles.append(smi)
            continue
        new_mol = replace_dummy_atoms(mol)
        new_smiles.append(Chem.MolToSmiles(new_mol))
    return new_smiles