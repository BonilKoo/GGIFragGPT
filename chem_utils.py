import re
import math

from rdkit import Chem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem.BRICS import reactionDefs

def build_allowed_mapping(reaction_defs):
    mapping = {}
    for level in reaction_defs:
        for atom_a, atom_b, _ in level:
            mapping.setdefault(atom_a, set()).add(atom_b)
            mapping.setdefault(atom_b, set()).add(atom_a)
    mapping.pop('7a', None)
    mapping.pop('7b', None)
    mapping.setdefault('7', set()).add('7')
    return mapping

ALLOWED_MAPPING = build_allowed_mapping(reactionDefs)

def build_allowed_connections(reaction_defs):
    """
    Returns a dictionary from the given reactionDefs where the key is a frozenset of two labels 
    and the value is the bond symbol connecting them.
    """
    allowed_connections = {}
    for group in reaction_defs:
        for a, b, bond in group:
            a = '7' if a in ['7a', '7b'] else a
            b = '7' if b in ['7a', '7b'] else b
            key = frozenset([a, b])
            allowed_connections[key] = bond
    return allowed_connections

ALLOWED_CONNECTIONS = build_allowed_connections(reactionDefs)

BOND_TYPE_MAPPING = {
    '-': Chem.BondType.SINGLE,
    '=': Chem.BondType.DOUBLE,
}

def assign_dummy_labels_to_dummy_atoms(mol, smiles):
    """
    Extracts dummy atom notations (e.g., "[5*]") from the input SMILES using regular expressions,
    and assigns them sequentially as a 'brics' property to dummy atoms (atomic number 0) in the Mol object.
    If the number of dummy atoms does not match, a warning message is printed.
    """
    # Extract all dummy atom notations (e.g., "[number*]") from the SMILES
    labels = re.findall(r'\[([0-9]+)\*\]', smiles)
    # Retrieve dummy atoms (atoms with atomic number 0) from the Mol object
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(labels) != len(dummy_atoms):
        print("Warning: dummy atom count mismatch. SMILES labels:", len(labels), "Mol dummy atoms:", len(dummy_atoms))
    # Assign the 'brics' property in order (if there are multiple dummy atoms, assign sequentially)
    for atom, label in zip(dummy_atoms, labels):
        atom.SetProp('brics', label)

def extract_dummy_info(mol):
    """
    Finds dummy atoms (atomic number 0) in the Mol object and returns a dictionary in the format:
    { label: (dummy_atom_index, neighbor_atom_index) },
    where the label is obtained from the 'brics' property or the mapping number, and the neighbor is the
    heavy (non-dummy) atom adjacent to the dummy atom.
    """
    info = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # dummy atom
            label = atom.GetProp('brics') if atom.HasProp('brics') else str(atom.GetAtomMapNum())
            heavy_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 0]
            if len(heavy_neighbors) != 1:
                raise ValueError("Dummy atom does not have exactly one heavy neighbor.")
            info[label] = (atom.GetIdx(), heavy_neighbors[0])
    return info

def combine_molecules(mol1, info1, mol2, info2, allowed_connections):
    """
    Takes two Mol objects (mol1, mol2) and their respective dummy atom information (info1, info2),
    finds a pair of dummy atoms that can be connected according to allowed_connections,
    removes the two dummy atoms, and adds an appropriate bond between the heavy atoms that were adjacent to them.
    Returns the combined Mol and the remaining dummy atom information.
    """
    connection_found = False
    for lab1, (dummy_idx1, heavy_idx1) in info1.items():
        for lab2, (dummy_idx2, heavy_idx2) in info2.items():
            key = frozenset([lab1, lab2])
            if key in allowed_connections:
                bond_symbol = allowed_connections[key]
                bond_type = BOND_TYPE_MAPPING[bond_symbol]
                connection_found = True
                break
        if connection_found:
            break
    if not connection_found:
        print(Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2))
        raise ValueError("No allowed connection found between the two fragments.")
    
    # The atom indices of mol2 have an offset when combined with mol1.
    offset = mol1.GetNumAtoms()
    heavy_idx2_new = heavy_idx2 + offset

    # Combine the two Mol objects.
    combo = Chem.CombineMols(mol1, mol2)
    rw_combo = Chem.RWMol(combo)
    # Add a bond between the two heavy atoms.
    rw_combo.AddBond(heavy_idx1, heavy_idx2_new, bond_type)
    
    # Remove the used dummy atoms (apply offset for dummy atoms from mol2).
    dummy_indices = [dummy_idx1, dummy_idx2 + offset]
    for idx in sorted(dummy_indices, reverse=True):
        rw_combo.RemoveAtom(idx)
    
    new_mol = rw_combo.GetMol()
    Chem.SanitizeMol(new_mol)
    
    # Combine the remaining dummy information from both molecules, excluding the two used labels.
    new_info = {}
    for lab, tup in info1.items():
        if lab != lab1:
            new_info[lab] = tup
    for lab, tup in info2.items():
        if lab != lab2:
            new_info[lab] = (tup[0] + offset, tup[1] + offset)
    return new_mol, new_info

def merge_fragment_smiles(current_smiles, current_fragments, finished):
    """
    For each index, takes the current SMILES (original molecule) and the SMILES of a fragment to be merged,
    and merges the two fragments according to allowed_connections rules.
    If finished is True for an index, the original SMILES is returned without merging.
    """
    merged_results = []
    for mol_smiles, frag_smiles, fin in zip(current_smiles, current_fragments, finished):
        if fin:
            merged_results.append(mol_smiles)
            continue
        try:
            # Create Mol objects with sanitize=False to prevent RDKit's internal reordering.
            mol = Chem.MolFromSmiles(mol_smiles, sanitize=False)
            frag = Chem.MolFromSmiles(frag_smiles, sanitize=False)
            # Extract dummy atom labels from the original SMILES and assign them to each dummy atom in the Mol.
            assign_dummy_labels_to_dummy_atoms(mol, mol_smiles)
            assign_dummy_labels_to_dummy_atoms(frag, frag_smiles)
            # Sanitize the Mol objects to clean them up.
            Chem.SanitizeMol(mol)
            Chem.SanitizeMol(frag)
            
            info1 = extract_dummy_info(mol)
            info2 = extract_dummy_info(frag)
            new_mol, _ = combine_molecules(mol, info1, frag, info2, ALLOWED_CONNECTIONS)
            new_smiles = Chem.MolToSmiles(new_mol)
            merged_results.append(new_smiles)
        except Exception as e:
            print("Merge failed:", e)
            merged_results.append(mol_smiles)
    return merged_results