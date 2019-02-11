# required features
# logP, TPSA, SAS, MW, graph2smiles


from os.path import join
import multiprocessing as mp

"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler, SequentialSampler
"""

from rdkit import Chem
from scipy.linalg import fractional_matrix_power
import numpy as np
from numpy.linalg import matrix_power
import pandas as pd

from utils_new import get_dir_files

list_num_atom = []

LIST_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']


LOGP_MEAN, LOGP_STD = 3.0475299537604004, 1.4508318866361838
# MR_MEAN, MR_STD = 1.983070758071883, 0.07702976853699765
TPSA_MEAN, TPSA_STD = 1.8082864863018322, 0.1832254436608209
MAX_LEN = 120


# index --> one - hot
def atom_feature(atom):
    return np.array(char_to_ix(atom.GetSymbol(), LIST_SYMBOLS) +
                    char_to_ix(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    char_to_ix(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    char_to_ix(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    char_to_ix(int(atom.GetIsAromatic()), [0, 1]))  # (40, 6, 5, 6, 2), 59


def char_to_ix(x, allowable_set):
    if x not in allowable_set:
        return [0]  # Unknown Atom Token
    return [allowable_set.index(x) + 1]


def normalize_adj(mx):
    """ Symmetry Normalization """
    rowsum = np.diag(np.array(mx.sum(1)))
    r_inv = fractional_matrix_power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.
    return r_inv.dot(mx).dot(r_inv)


def mol2graph(smi):
    mol = Chem.MolFromSmiles(smi)

    X = np.zeros((MAX_LEN, 5), dtype=np.uint8)
    A = np.zeros((MAX_LEN, MAX_LEN), dtype=np.uint8)

    temp_A = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.uint8, copy=False)[:MAX_LEN, :MAX_LEN]
    num_atom = temp_A.shape[0]
    A[:num_atom, :num_atom] = temp_A + np.eye(temp_A.shape[0], dtype=np.uint8)

    for i, atom in enumerate(mol.GetAtoms()):
        feature = atom_feature(atom)
        X[i, :] = feature
        if i + 1 >= num_atom: break

    return X, A


def preprocess_df(smiles, num_worker):
    with mp.Pool(processes=num_worker) as pool:
        mols = pool.map(mol2graph, smiles)
    X, A = list(zip(*mols))
    X = np.array(X, dtype=np.uint8)
    A = np.array(A, dtype=np.uint8)
    return X, A

"""
class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last=False, shuffle_batch=False):

        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batch = shuffle_batch

    def __iter__(self):
        batch = list()
        mini_batch = list()
        for idx in self.sampler:
            mini_batch.append(idx)
            if len(mini_batch) == self.batch_size:
                batch.append(mini_batch)
                mini_batch = []
        if len(mini_batch) > 0 and not self.drop_last:
            batch.append(mini_batch)

        if self.shuffle_batch:
            return iter(np.random.permutation(batch))
        else:
            return iter(batch)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class zincDataset(Dataset):
    def __init__(self, data_path, filename, num_worker, save_cache=True, labels=['logP', 'sas', 'tpsa']):
        # Make Label Index
        label2idx = {'logP': 0, 'sas': 1, 'tpsa': 2}
        self.label_idx = np.array([label2idx[label] for label in labels])

        # Find whether cache is exist
        files = get_dir_files(data_path)
        cache_name = filename + '.npz'
        if cache_name in files:
            print("Cache Found. Loading Preprocessed Data from {}...".format(cache_name))
            temp = np.load(join(data_path, cache_name))
            self.X = temp['X']
            self.A = temp['A']
            self.C = temp['C']
            self.L = temp['L']
            print("Loading Preprocessed Data Complete!".format(cache_name))

        else:
            print("Cache Not Found. Loading Dataset from {}...".format(filename))
            # Load data from raw dataset
            self.data = pd.read_csv(join(data_path, filename))
            self.data = self.data.reset_index()
            print("Dataset Loading Complete")

            # Mean & Std Normalize of molecular property
            self.data.logP = (self.data.logP - LOGP_MEAN) / LOGP_STD
            self.data.sas = np.log10(self.data.sas + 1)
            self.data.sas = (self.data.sas - MR_MEAN) / MR_STD
            self.data.tpsa = np.log10(self.data.tpsa + 1)
            self.data.tpsa = (self.data.tpsa - TPSA_MEAN) / TPSA_STD
            print("Molecular Property Normalization Complete!")

            # Get Property Matrix
            # self.C = self.data[['logP', 'mr', 'tpsa', 'sa']].values
            self.C = self.data[['logP', 'sas', 'tpsa']].values
            self.L = self.data['length']
            smiles = self.data.smile.values
            del self.data

            # Convert smiles to Graph
            print("Converting Smiles to Graph...")
            self.X, self.A = preprocess_df(smiles, num_worker)
            del smiles
            print("Converting Smiles to Graph Complete!")

            # Save Preprocessed Data
            if save_cache:
                print("Saving Preprocessed Data to {}...".format(cache_name))
                np.savez_compressed(join(data_path, filename), X=self.X, A=self.A, C=self.C, L=self.L)
                print("Saving Preprocessed Data Complete!")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        A = self.A[index]
        C = self.C[index, self.label_idx]
        L = self.L[index]
        return X, A, C, L


if __name__ == '__main__':
    # a = result file path
    dataset = zincDataset(a)
"""