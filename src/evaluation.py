import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score.sascorer import calculateScore

from fcd_torch import FCD

def cal_validity(smi_list):
    val = 0
    valid_smi_list = []
    for idx, smi in enumerate(smi_list):
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            val += 1
            valid_smi_list.append(smi)
        except:
            print(f"Invalid SMILES: {smi}")
            pass
    validity = val / len(smi_list)
    return validity, valid_smi_list

def cal_uniqueness(smi_list):
    uniq_smi_list = list(set(smi_list))
    uniq = len(uniq_smi_list) / len(smi_list)
    return uniq, uniq_smi_list

def cal_internal_diversity(smi_list):
    # Convert SMILES strings to Molecule objects
    mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
    # Precompute fingerprints from Molecule objects
    fps = [Chem.RDKFingerprint(mol) for mol in mol_list]
    
    total_diversity = 0
    count = 0
    n = len(fps)
    # For each fingerprint, compare with all subsequent fingerprints using BulkTanimotoSimilarity
    for i in range(n-1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        # Calculate diversity as 1 - similarity and accumulate the total diversity
        total_diversity += sum(1 - sim for sim in sims)
        count += len(sims)
    
    # Return the average diversity across all pairs
    return total_diversity / count

def cal_QEDscore(smi_list):
    QED_scores = [QED.default(Chem.MolFromSmiles(smi)) for smi in smi_list]
    return QED_scores

def cal_SAscore(smi_list):
    SA_scores = [calculateScore(Chem.MolFromSmiles(smi)) for smi in smi_list]
    return SA_scores

def cal_novelty(smi_list, ref_smi_list):
    smi_set = set(smi_list)
    return 1 - len(set(ref_smi_list).intersection(smi_set)) / len(smi_set)

def cal_FCD(smi_list, ref_smi_list, device):
    fcd = FCD(device=device, n_jobs=8)
    fcd_score = fcd(gen=smi_list, ref=ref_smi_list)
    return fcd_score

def evaluate(smi_list, ref_smi_list, device):
    validity, smi_valid = cal_validity(smi_list)
    print(f'Validity: {validity}')

    novelty = cal_novelty(smi_valid, ref_smi_list)
    print(f'Novelty: {novelty}')
    
    uniqueness, smi_valid_uniq = cal_uniqueness(smi_valid)
    print(f'Uniqueness: {uniqueness}')
    
    internal_diversity = cal_internal_diversity(smi_valid_uniq)
    print(f'Internal_diversity: {internal_diversity}')

    fcd_score = cal_FCD(smi_valid, ref_smi_list, device)
    print(f'Fréchet ChemNet Distance: {fcd_score}')
    
    QED_scores = cal_QEDscore(smi_valid_uniq)
    QED_mean = np.mean(QED_scores)
    print(f'QED: {QED_mean}')
    
    SA_scores = cal_SAscore(smi_valid_uniq)
    SA_mean = np.mean(SA_scores)
    print(f'SA: {SA_mean}')
    
    return {
        'validity': validity,
        'novelty': novelty,
        'uniqueness': uniqueness,
        'internal diversity': internal_diversity,
        'Fréchet ChemNet Distance': fcd_score,
        'QED': QED_mean,
        'SA': SA_mean
           }